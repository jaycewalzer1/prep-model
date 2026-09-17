"""Stage 2: the deterministic stratified model.

Expected counts across housing x HIV stage x care state. No individuals, no
random draws, and a batch dimension so thousands of parameter sets run at once.
This is the model used for debugging, for calibration, and for broad parameter
exploration; the microsimulation is the model used for results.

It is deliberately coarser than Stage 3 in one respect: exposure groups are
collapsed into a single population-average exposure rate. That makes the
deterministic hazard an approximation to the mixing model in
``transmission.py``, so calibrated values transported into Stage 3 are checked
against Stage 3's own outputs rather than assumed to carry over exactly.

The number of independently fitted parameters is kept below the number of
calibration targets. ``calibration.py`` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .params import Params
from .states import CareState, HivStage, Housing, N_EXPOSURE_GROUPS

N_HOUSING, N_STAGE, N_CARE = 3, 4, 4


def _p(rate: torch.Tensor, dt: float) -> torch.Tensor:
    return 1.0 - torch.exp(-rate.clamp_min(0.0) * dt)


@dataclass
class DeterministicOutputs:
    diagnosed_prevalence: torch.Tensor
    annual_diagnoses_per_1000: torch.Tensor
    suppressed_among_diagnosed: torch.Tensor
    share_unsheltered: torch.Tensor
    prevalence: torch.Tensor
    incidence_per_100py: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.__dict__)


def _batch(values: list[float] | torch.Tensor, b: int) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float64)
    return t.expand(b).clone() if t.dim() == 0 else t


def _scalar_params(p: Params, names: list[str], batch: dict[str, torch.Tensor],
                   b: int) -> dict[str, torch.Tensor]:
    """Per-batch values: from ``batch`` when supplied, otherwise the registry value."""
    out = {}
    for n in names:
        out[n] = batch[n].to(torch.float64) if n in batch else torch.full((b,), p[n],
                                                                         dtype=torch.float64)
    return out


FREE = [
    "lambda_sex_scale", "lambda_inj_scale", "external_hazard_sex_annual",
    "external_hazard_inj_annual", "testing_rate_annual", "art_init_rate_annual",
    "disengage_rate_annual", "reengage_rate_annual", "baseline_undiagnosed_fraction",
    "baseline_hiv_prevalence",
]


def run(p: Params, n_years: float, dt: float, n_people: float = 1000.0,
        batch: dict[str, torch.Tensor] | None = None) -> DeterministicOutputs:
    """Integrate usual care forward and return the calibration summaries."""
    batch = batch or {}
    b = int(next(iter(batch.values())).shape[0]) if batch else 1
    q = _scalar_params(p, FREE, batch, b)

    # -- initial condition -----------------------------------------------------
    x = torch.zeros(b, N_HOUSING, N_STAGE, N_CARE, dtype=torch.float64)
    housed = max(0.0, 1.0 - p["frac_unsheltered"] - p["frac_sheltered"])
    housing_shares = torch.tensor([p["frac_unsheltered"], p["frac_sheltered"], housed],
                                  dtype=torch.float64)
    housing_shares = housing_shares / housing_shares.sum()

    prev = q["baseline_hiv_prevalence"]
    undx = q["baseline_undiagnosed_fraction"]
    art = (1 - undx) * p["baseline_art_among_diagnosed"]
    sup = art * p["baseline_suppressed_among_art"]
    care_shares = torch.stack([undx, 1 - undx - art, art - sup, sup], dim=1).clamp_min(0.0)
    care_shares = care_shares / care_shares.sum(dim=1, keepdim=True)

    for h in range(N_HOUSING):
        share = float(housing_shares[h]) * n_people
        x[:, h, int(HivStage.SUSCEPTIBLE), int(CareState.UNDIAGNOSED)] = share * (1 - prev)
        for k in range(N_CARE):
            x[:, h, int(HivStage.CHRONIC), k] = share * prev * care_shares[:, k] * 0.88
            x[:, h, int(HivStage.ADVANCED), k] = share * prev * care_shares[:, k] * 0.12

    # -- fixed structure -------------------------------------------------------
    exposure = torch.tensor(p.vector("exposure_rate_sex_g", N_EXPOSURE_GROUPS), dtype=torch.float64)
    shares = torch.tensor(p.vector("exposure_share_g", N_EXPOSURE_GROUPS), dtype=torch.float64)
    shares = shares / shares.sum()
    mean_exposure = float((exposure * shares).sum())
    frac_sharers = float((torch.tensor(p.vector("frac_injects_g", N_EXPOSURE_GROUPS),
                                       dtype=torch.float64) * shares).sum())

    w_sex = torch.zeros(N_STAGE, N_CARE, dtype=torch.float64)
    w_inj = torch.zeros(N_STAGE, N_CARE, dtype=torch.float64)
    for s, base in ((int(HivStage.ACUTE), p["infectiousness_acute"]),
                    (int(HivStage.CHRONIC), 1.0),
                    (int(HivStage.ADVANCED), p["infectiousness_advanced"])):
        w_sex[s, :] = base
        w_inj[s, :] = base
    w_sex[:, int(CareState.ART_SUPPRESSED)] *= p["infectiousness_suppressed_sexual"]
    w_inj[:, int(CareState.ART_SUPPRESSED)] *= p["infectiousness_suppressed_injection"]

    bg_mort = torch.tensor([p["bg_mortality_45_59"] * p["mortality_multiplier_unsheltered"],
                            p["bg_mortality_45_59"] * p["mortality_multiplier_sheltered"],
                            p["bg_mortality_45_59"]], dtype=torch.float64)
    hiv_mort = torch.zeros(N_STAGE, N_CARE, dtype=torch.float64)
    hiv_mort[int(HivStage.CHRONIC), :] = p["chronic_mortality_excess"]
    hiv_mort[int(HivStage.ADVANCED), :] = p["advanced_mortality_rate"]
    hiv_mort[:, int(CareState.ART_UNSUPPRESSED)] *= p["art_mortality_multiplier"] * 2.0
    hiv_mort[:, int(CareState.ART_SUPPRESSED)] *= p["art_mortality_multiplier"]

    housing_rates = torch.zeros(N_HOUSING, N_HOUSING, dtype=torch.float64)
    housing_rates[Housing.UNSHELTERED, Housing.SHELTERED] = p["housing_rate_u_to_s"]
    housing_rates[Housing.UNSHELTERED, Housing.HOUSED] = p["housing_rate_u_to_h"]
    housing_rates[Housing.SHELTERED, Housing.UNSHELTERED] = p["housing_rate_s_to_u"]
    housing_rates[Housing.SHELTERED, Housing.HOUSED] = p["housing_rate_s_to_h"]
    housing_rates[Housing.HOUSED, Housing.UNSHELTERED] = p["housing_rate_h_to_u"]
    housing_rates[Housing.HOUSED, Housing.SHELTERED] = p["housing_rate_h_to_s"]

    n_steps = int(round(n_years / dt))
    diagnoses_window: list[torch.Tensor] = []
    person_years_window: list[torch.Tensor] = []
    infections_total = torch.zeros(b, dtype=torch.float64)
    person_years_total = torch.zeros(b, dtype=torch.float64)
    inflow = p["inflow_rate_per_year"] * n_people
    oral_cover = p["baseline_oral_prep_use"] * p["oral_efficacy_sexual"]

    for step in range(n_steps):
        n = x.sum(dim=(1, 2, 3)).clamp_min(1e-9)
        person_years_total += n * dt

        prev_sex = (x * w_sex).sum(dim=(1, 2, 3)) / n
        prev_inj = (x * w_inj).sum(dim=(1, 2, 3)) / n
        lam = q["lambda_sex_scale"] * mean_exposure * prev_sex + q["external_hazard_sex_annual"]
        lam = lam + frac_sharers * (q["lambda_inj_scale"] * prev_inj
                                    + q["external_hazard_inj_annual"])
        lam = lam * (1.0 - oral_cover)

        s_idx, a_idx, c_idx, adv_idx = (int(HivStage.SUSCEPTIBLE), int(HivStage.ACUTE),
                                        int(HivStage.CHRONIC), int(HivStage.ADVANCED))
        new_inf = x[:, :, s_idx, int(CareState.UNDIAGNOSED)] * \
            _p(lam, dt).unsqueeze(1)
        x[:, :, s_idx, int(CareState.UNDIAGNOSED)] -= new_inf
        x[:, :, a_idx, int(CareState.UNDIAGNOSED)] += new_inf
        infections_total += new_inf.sum(dim=1)

        # Stage transitions.
        to_chronic = x[:, :, a_idx, :] * _p(torch.tensor(1.0 / p["acute_duration_years"],
                                                         dtype=torch.float64), dt)
        x[:, :, a_idx, :] -= to_chronic
        x[:, :, c_idx, :] += to_chronic

        prog = x[:, :, c_idx, :] * _p(torch.tensor(p["chronic_to_advanced_rate"],
                                                   dtype=torch.float64), dt)
        prog[:, :, int(CareState.ART_SUPPRESSED)] = 0.0
        x[:, :, c_idx, :] -= prog
        x[:, :, adv_idx, :] += prog

        back = x[:, :, adv_idx, int(CareState.ART_SUPPRESSED)] * \
            _p(torch.tensor(p["art_regression_rate"], dtype=torch.float64), dt)
        x[:, :, adv_idx, int(CareState.ART_SUPPRESSED)] -= back
        x[:, :, c_idx, int(CareState.ART_SUPPRESSED)] += back

        # Care cascade. Only infected compartments move.
        test_rate = q["testing_rate_annual"].view(b, 1, 1)
        undx_pool = x[:, :, 1:, int(CareState.UNDIAGNOSED)]
        mult = torch.ones(1, 1, N_STAGE - 1, dtype=torch.float64)
        mult[0, 0, adv_idx - 1] = p["testing_rate_advanced_multiplier"]
        dx = undx_pool * _p(test_rate * mult, dt)
        x[:, :, 1:, int(CareState.UNDIAGNOSED)] -= dx
        x[:, :, 1:, int(CareState.DIAGNOSED_NO_ART)] += dx
        diagnoses_window.append(dx.sum(dim=(1, 2)))
        person_years_window.append(n * dt)

        start = x[:, :, 1:, int(CareState.DIAGNOSED_NO_ART)] * \
            _p(q["art_init_rate_annual"].view(b, 1, 1), dt)
        x[:, :, 1:, int(CareState.DIAGNOSED_NO_ART)] -= start
        x[:, :, 1:, int(CareState.ART_UNSUPPRESSED)] += start

        sup_flow = x[:, :, 1:, int(CareState.ART_UNSUPPRESSED)] * \
            _p(torch.tensor(p["suppression_rate_on_art"], dtype=torch.float64), dt)
        x[:, :, 1:, int(CareState.ART_UNSUPPRESSED)] -= sup_flow
        x[:, :, 1:, int(CareState.ART_SUPPRESSED)] += sup_flow

        fail = x[:, :, 1:, int(CareState.ART_SUPPRESSED)] * \
            _p(torch.tensor(p["art_failure_rate"], dtype=torch.float64), dt)
        x[:, :, 1:, int(CareState.ART_SUPPRESSED)] -= fail
        x[:, :, 1:, int(CareState.ART_UNSUPPRESSED)] += fail

        dis_p = _p(q["disengage_rate_annual"].view(b, 1, 1), dt)
        for k in (int(CareState.ART_UNSUPPRESSED), int(CareState.ART_SUPPRESSED)):
            left = x[:, :, 1:, k] * dis_p
            x[:, :, 1:, k] -= left
            x[:, :, 1:, int(CareState.DIAGNOSED_NO_ART)] += left

        back_in = x[:, :, 1:, int(CareState.DIAGNOSED_NO_ART)] * \
            _p(q["reengage_rate_annual"].view(b, 1, 1), dt)
        x[:, :, 1:, int(CareState.DIAGNOSED_NO_ART)] -= back_in
        x[:, :, 1:, int(CareState.ART_UNSUPPRESSED)] += back_in

        # Exits: death and migration as one competing-hazard step.
        total_exit = bg_mort.view(1, N_HOUSING, 1, 1) + hiv_mort.view(1, 1, N_STAGE, N_CARE) + \
            p["migration_rate_per_year"]
        x = x * (1.0 - _p(total_exit, dt))

        # Housing moves.
        moved = torch.zeros_like(x)
        for origin in range(N_HOUSING):
            for dest in range(N_HOUSING):
                if origin == dest or housing_rates[origin, dest] == 0:
                    continue
                flow = x[:, origin] * _p(housing_rates[origin, dest], dt)
                moved[:, origin] -= flow
                moved[:, dest] += flow
        x = x + moved

        # Exogenous inflow at the baseline composition.
        for h in range(N_HOUSING):
            share = float(housing_shares[h]) * inflow * dt
            x[:, h, s_idx, int(CareState.UNDIAGNOSED)] += share * (1 - prev)
            for k in range(N_CARE):
                x[:, h, c_idx, k] += share * prev * care_shares[:, k] * 0.88
                x[:, h, adv_idx, k] += share * prev * care_shares[:, k] * 0.12

    n = x.sum(dim=(1, 2, 3)).clamp_min(1e-9)
    infected = x[:, :, 1:, :].sum(dim=(1, 2, 3))
    diagnosed = x[:, :, 1:, 1:].sum(dim=(1, 2, 3))
    suppressed = x[:, :, 1:, int(CareState.ART_SUPPRESSED)].sum(dim=(1, 2))
    window = max(1, int(round(1.0 / dt)))
    recent_dx = torch.stack(diagnoses_window[-window:]).sum(dim=0)
    recent_py = torch.stack(person_years_window[-window:]).sum(dim=0)

    return DeterministicOutputs(
        diagnosed_prevalence=diagnosed / n,
        annual_diagnoses_per_1000=recent_dx / recent_py.clamp_min(1e-9) * 1000.0,
        suppressed_among_diagnosed=suppressed / diagnosed.clamp_min(1e-9),
        share_unsheltered=x[:, int(Housing.UNSHELTERED)].sum(dim=(1, 2)) / n,
        prevalence=infected / n,
        incidence_per_100py=infections_total / person_years_total.clamp_min(1e-9) * 100.0,
    )
