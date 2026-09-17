"""What would it take to drive HIV incidence in this population to zero, and what
would it cost?

Two separate questions get confused when people say "eradicate it". One is how
much programme you would have to buy. The other is whether buying all of it would
be enough. This module answers the second one first, because if there is a floor
the programme cannot reach, the cost of reaching it is not a number.

The ladder raises outreach reach and delivery capacity together, from the
configured programme up to a saturated one in which capacity never binds and
essentially everyone reachable is reached. At the top rung the residual infections
are decomposed against three structural assumptions the model makes:

- an **external hazard** that does not depend on the modelled population's
  prevalence, so no amount of internal coverage removes it;
- **zero efficacy against the equipment-sharing route** in the primary analysis,
  so people who acquire that way are unprotected however well they are served;
- people who **refuse, never initiate, or lapse**, which saturating capacity does
  not fix because the barrier is not capacity.

Each is turned off in turn at full coverage. The gap that remains when all three
are off is what the delivery system itself is responsible for.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import simulate
from .config import Config, load_config
from .params import Params, Registry
from .population import build_cohort

SATURATED = {
    "capacity_contacts_per_1000_per_week": 10_000.0,
    "capacity_initiations_per_1000_per_week": 1_000.0,
}
"""Capacity so large it can never bind. The same values the base configuration
already uses for its ``unconstrained_capacity`` structural scenario."""


@dataclass
class Rung:
    label: str
    reach_multiple: float
    infections: float
    person_years: float
    cost: float
    doses: float
    on_program_share: float

    @property
    def incidence_per_100py(self) -> float:
        return 100.0 * self.infections / max(1.0, self.person_years)


@dataclass
class Floor:
    label: str
    infections: float
    person_years: float

    @property
    def incidence_per_100py(self) -> float:
        return 100.0 * self.infections / max(1.0, self.person_years)


@dataclass
class EliminationResult:
    arm_id: str
    reference_infections: float
    reference_person_years: float
    reference_cost: float
    rungs: list[Rung] = field(default_factory=list)
    floors: list[Floor] = field(default_factory=list)

    @property
    def reference_incidence(self) -> float:
        return 100.0 * self.reference_infections / max(1.0, self.reference_person_years)

    def to_dict(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "reference_infections": self.reference_infections,
            "reference_incidence_per_100py": self.reference_incidence,
            "rungs": [{**r.__dict__, "incidence_per_100py": r.incidence_per_100py}
                      for r in self.rungs],
            "floors": [{**f.__dict__, "incidence_per_100py": f.incidence_per_100py}
                       for f in self.floors],
        }


def _mean_over_replicates(cfg: Config, p: Params, arm_id: str, reps: int) -> simulate.ArmResult:
    """One arm at one parameter set. Replicates are averaged by the caller."""
    return simulate.run_arm(cfg, p, cfg.arm(arm_id), replicate=0)


def _run(cfg: Config, p: Params, arm_id: str, reps: int) -> tuple[float, float, float, float, float]:
    """Infections, person-years, cost, doses and mean on-programme share, averaged."""
    cats = cfg.perspectives[cfg.primary_perspective]
    inf = py = cost = doses = cov = 0.0
    for r in range(reps):
        res = simulate.run_arm(cfg, p, cfg.arm(arm_id), replicate=r)
        inf += res.epi["infections"]
        py += res.epi["person_years"]
        cost += res.cost(cats)
        doses += res.counters.doses
        cov += (sum(res.trace["coverage"]) / max(1, len(res.trace["coverage"])))
    return inf / reps, py / reps, cost / reps, doses / reps, cov / reps


def ladder(cfg: Config, p: Params, arm_id: str = "C",
           multiples: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
           reps: int = 1) -> EliminationResult:
    """Raise reach and capacity together and watch where incidence stops falling."""
    ref_inf, ref_py, ref_cost, _, _ = _run(cfg, p, cfg.reference_arm, reps)
    out = EliminationResult(arm_id=arm_id, reference_infections=ref_inf,
                            reference_person_years=ref_py, reference_cost=ref_cost)

    base_reach = p["reach_rate_annual_lowbarrier"]
    for m in multiples:
        pm = p.with_overrides({**SATURATED, "reach_rate_annual_lowbarrier": base_reach * m})
        inf, py, cost, doses, cov = _run(cfg, pm, arm_id, reps)
        out.rungs.append(Rung(label=f"reach x{m:g}", reach_multiple=m, infections=inf,
                              person_years=py, cost=cost, doses=doses,
                              on_program_share=cov))

    # At the top rung, remove each structural barrier in turn.
    top = p.with_overrides({**SATURATED,
                            "reach_rate_annual_lowbarrier": base_reach * multiples[-1]})
    variants = {
        "saturated programme": {},
        "+ no external hazard": {"external_hazard_sex_annual": 0.0,
                                 "external_hazard_inj_annual": 0.0},
        "+ efficacy on the injection route": {"injectable_efficacy_injection_route": 0.70},
        "+ nobody refuses or fails to start": {"refusal_permanent_prob": 0.0,
                                               "accept_prob_lowbarrier": 1.0,
                                               "initiate_given_accept_lowbarrier": 1.0},
        "+ perfect retention": {"attendance_prob_lowbarrier": 1.0,
                               "ae_discontinuation_rate_annual": 0.0},
    }
    cumulative: dict[str, float] = {}
    for label, ov in variants.items():
        cumulative.update(ov)
        inf, py, _, _, _ = _run(cfg, top.with_overrides(cumulative), arm_id, reps)
        out.floors.append(Floor(label=label, infections=inf, person_years=py))
    return out


def render(out: EliminationResult) -> str:
    lines: list[str] = []
    lines.append(f"usual care: {out.reference_infections:,.0f} infections, "
                 f"{out.reference_incidence:.3f} per 100 person-years")
    lines.append("")
    lines.append("Raising outreach reach with capacity never binding:")
    lines.append("")
    lines.append("| programme | incidence /100py | infections | averted vs usual care | "
                 "mean coverage | doses | incremental cost | per infection averted |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in out.rungs:
        av = out.reference_infections - r.infections
        # Against usual care, not the arm's own gross spend: most of that is the
        # background housing and non-HIV care that appears identically in every arm.
        inc = r.cost - out.reference_cost
        cpa = "n/a" if av <= 0 else f"${inc / av:,.0f}"
        lines.append(f"| {r.label} | {r.incidence_per_100py:.3f} | {r.infections:,.0f} | "
                     f"{av:,.0f} | {r.on_program_share:.1%} | {r.doses:,.0f} | "
                     f"${inc:,.0f} | {cpa} |")
    lines.append("")
    lines.append("What is left at the top rung, removing one barrier at a time "
                 "(each row keeps the ones above it):")
    lines.append("")
    lines.append("| assumption removed | incidence /100py | infections |")
    lines.append("|---|---|---|")
    for f in out.floors:
        lines.append(f"| {f.label} | {f.incidence_per_100py:.3f} | {f.infections:,.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--arm", default="C")
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    p = Registry.load().resolve(cfg.price_year, overrides=cfg.overrides)
    out = ladder(cfg, p, a.arm, reps=a.replicates)
    text = render(out)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out.to_dict(), indent=2))
        print(f"\nwritten {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
