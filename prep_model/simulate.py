"""Stage 3: the tensorized stochastic microsimulation.

One arm, one replicate, one call. The population is built once from the seed
and cloned per arm, so arms start identical and stay paired by construction.

Ordering inside a step is fixed and deliberate. Health-state costs and health
accrue for the state a person was in at the start of the week. Then delivery,
then acquisition, then disease and care transitions, then housing, then exits.
Exits last means a person cannot die and also initiate PrEP in the same week;
death and out-migration are resolved together as competing hazards so they can
never both happen either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import delivery, economics, housing, natural_history, transmission
from .config import Arm, Config
from .delivery import DeliveryCounters
from .economics import Ledger
from .params import Params
from .population import Cohort, build_cohort
from .prep import protection, regimen_for
from .states import CareState, HivStage, Housing, Product


@dataclass
class ArmResult:
    arm_id: str
    arm_name: str
    replicate: int
    ledger: Ledger
    counters: DeliveryCounters
    epi: dict[str, float]
    annual: dict[str, list[float]]
    trace: dict[str, list[float]]
    accounting: dict[str, float]
    calibration_outputs: dict[str, float]

    def cost(self, categories: list[str]) -> float:
        return self.ledger.total_discounted(categories)

    @property
    def qalys(self) -> float:
        return self.ledger.total_qalys()


def _year_index(step: int, dt: float, n_years: int) -> int:
    return min(int(step * dt), n_years - 1)


def run_arm(cfg: Config, p: Params, arm: Arm, replicate: int = 0,
            baseline: Cohort | None = None,
            frozen_prevalence: bool = False) -> ArmResult:
    seed = cfg.seed + 10_000 * replicate
    dt = cfg.dt
    n_steps = cfg.n_steps
    n_years = max(1, int(round(cfg.horizon_years)))

    c = (baseline.clone() if baseline is not None
         else build_cohort(seed, cfg.n_individuals, n_steps, dt, p))
    ledger = Ledger(n_steps=n_steps, dt=dt, discount_rate=cfg.discount_rate, n_years=n_years)
    counters = DeliveryCounters()
    settings = delivery.settings_for(arm, p)
    regimen = regimen_for(arm.product, p)

    frozen = transmission.baseline_prevalence(c, p, c.active(0)) if frozen_prevalence else None

    annual = {k: [0.0] * n_years for k in
              ("infections", "hiv_deaths", "deaths", "doses", "initiations", "contacts",
               "diagnoses", "person_years", "person_years_on_program")}
    trace = {k: [] for k in ("cumulative_infections", "on_program", "active", "coverage")}

    totals = {"infections": 0, "deaths": 0, "hiv_deaths": 0, "migrations": 0,
              "background_diagnoses": 0, "art_initiations": 0, "disengagements": 0,
              "progressed_to_advanced": 0, "housing_moves": 0}
    cumulative_infections = 0

    for step in range(n_steps):
        active = c.active(step)
        n_active = int(active.sum())
        year = _year_index(step, dt, n_years)

        economics.accrue_step(ledger, c, active, p, step, dt)
        delivery.accrue_background_prevention(ledger, c, active, p, step, dt)
        delivery.accrue_fixed_program_cost(ledger, arm, n_active, p, step, dt)
        annual["person_years"][year] += n_active * dt

        if settings is not None:
            k = delivery.step_delivery(c, arm, settings, regimen, ledger, seed, step, dt, p, active)
            counters.add(k)
            annual["doses"][year] += k.doses
            annual["initiations"][year] += k.initiations
            annual["contacts"][year] += k.contacts
            annual["diagnoses"][year] += k.program_diagnoses

        e_sex, e_inj = protection(c, regimen, p)
        inf = transmission.step_infections(c, seed, step, dt, p, active, e_sex, e_inj, frozen)
        totals["infections"] += inf["infections"]
        cumulative_infections += inf["infections"]
        annual["infections"][year] += inf["infections"]

        dis = natural_history.step_disease(c, seed, step, dt, p, active)
        for key, v in dis.items():
            totals[key] = totals.get(key, 0) + v
        annual["diagnoses"][year] += dis["background_diagnoses"]

        mv = housing.step_housing(c, seed, step, dt, p, active)
        totals["housing_moves"] += mv["housing_moves"]

        ex = natural_history.step_exits(c, seed, step, dt, p, active)
        totals["deaths"] += ex["deaths"]
        totals["hiv_deaths"] += ex["hiv_deaths"]
        totals["migrations"] += ex["migrations"]
        annual["deaths"][year] += ex["deaths"]
        annual["hiv_deaths"][year] += ex["hiv_deaths"]

        on_prog = active & c.on_program
        c.person_years_on_program = torch.where(on_prog,
                                                c.person_years_on_program + dt,
                                                c.person_years_on_program)
        annual["person_years_on_program"][year] += int(on_prog.sum()) * dt
        c.age = torch.where(active, c.age + dt, c.age)
        c.weeks_since_dose = torch.where(active & c.on_program,
                                         c.weeks_since_dose + cfg.step_weeks,
                                         c.weeks_since_dose)

        trace["cumulative_infections"].append(float(cumulative_infections))
        trace["on_program"].append(float(on_prog.sum()))
        trace["active"].append(float(n_active))
        trace["coverage"].append(float(on_prog.sum()) / max(1, n_active))

    final_active = c.active(n_steps - 1)
    terminal_info = {}
    if cfg.terminal_value:
        terminal_info = economics.accrue_terminal(ledger, c, final_active, p, n_steps)

    with_hiv = final_active & (c.hiv_stage != int(HivStage.SUSCEPTIBLE))
    diagnosed = with_hiv & (c.care_state >= int(CareState.DIAGNOSED_NO_ART))
    suppressed = with_hiv & (c.care_state == int(CareState.ART_SUPPRESSED))
    n_final = max(1, int(final_active.sum()))

    epi = {
        "infections": float(totals["infections"]),
        "deaths": float(totals["deaths"]),
        "hiv_deaths": float(totals["hiv_deaths"]),
        "migrations": float(totals["migrations"]),
        "person_years": float(sum(annual["person_years"])),
        "person_years_on_program": float(sum(annual["person_years_on_program"])),
        "life_years_discounted": ledger.life_years,
        "qalys_discounted": ledger.total_qalys(),
        "final_population": float(final_active.sum()),
        "final_prevalence": float(with_hiv.sum()) / n_final,
        "ever_initiated": float(c.ever_initiated.sum()),
        "switchers_from_oral": float(c.switched_from_oral.sum()),
        **{k: float(v) for k, v in totals.items()},
        **terminal_info,
    }

    accounting = {
        "allocated": float(c.n),
        "entered": float((c.entry_step < n_steps).sum()),
        "deaths": float(totals["deaths"]),
        "migrations": float(totals["migrations"]),
        "alive_and_resident_at_end": float(final_active.sum()),
        "not_yet_entered": float((c.entry_step >= n_steps).sum()),
    }

    calibration_outputs = {
        "diagnosed_prevalence": float(diagnosed.sum()) / n_final,
        "annual_diagnoses_per_1000": (sum(annual["diagnoses"][-3:]) / 3.0) /
                                     max(1.0, sum(annual["person_years"][-3:]) / 3.0) * 1000.0,
        "suppressed_among_diagnosed": float(suppressed.sum()) / max(1, int(diagnosed.sum())),
        "share_unsheltered": float((final_active & (c.housing == int(Housing.UNSHELTERED))).sum()) / n_final,
    }

    return ArmResult(
        arm_id=arm.id, arm_name=arm.name, replicate=replicate, ledger=ledger,
        counters=counters, epi=epi, annual=annual, trace=trace,
        accounting=accounting, calibration_outputs=calibration_outputs,
    )


def run_all_arms(cfg: Config, p: Params, replicate: int = 0,
                 frozen_prevalence: bool = False) -> dict[str, ArmResult]:
    """Every arm from one shared starting population and one shared seed."""
    seed = cfg.seed + 10_000 * replicate
    baseline = build_cohort(seed, cfg.n_individuals, cfg.n_steps, cfg.dt, p)
    return {arm.id: run_arm(cfg, p, arm, replicate, baseline, frozen_prevalence)
            for arm in cfg.arms}
