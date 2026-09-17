"""Route-specific acquisition hazards with an evolving infectious prevalence.

Mixing is group-based. A dense all-to-all network buys nothing here, and an
explicit sparse partnership network would need partnership data the project
does not have; if that data arrives, this module is the place it would go.

Infectious prevalence is recomputed every step from the live population, so
infections and treatment entries feed back into everyone's hazard. Freezing it
(``frozen_prevalence``) turns the model into a direct-protection-only model,
which is offered as a structural sensitivity analysis and clearly labelled,
because a frozen model cannot speak about secondary transmission at all.

Units. ``exposure_rate_sex_g*`` are dimensionless relative rates. The product
``lambda_sex_scale * exposure_rate * sum_h(mixing * infectious_prevalence)`` is
an annual hazard per susceptible person. Infectiousness weights are relative to
untreated chronic infection, which is fixed at 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .natural_history import infectiousness_injection, infectiousness_sexual
from .params import Params
from .population import Cohort
from .rng import Stream, uniform
from .states import HivStage, N_EXPOSURE_GROUPS


@dataclass
class MixingModel:
    """``mixing[g, h]`` is the share of group g's contacts that are with group h."""

    matrix: torch.Tensor

    @classmethod
    def build(cls, p: Params, group_weights: torch.Tensor) -> "MixingModel":
        a = p["mixing_assortativity"]
        n = group_weights.shape[0]
        proportional = group_weights / group_weights.sum().clamp_min(1e-12)
        m = a * torch.eye(n, dtype=torch.float64) + (1 - a) * proportional.unsqueeze(0).expand(n, n)
        return cls(m / m.sum(dim=1, keepdim=True).clamp_min(1e-12))


def _group_weighted_prevalence(group: torch.Tensor, weight: torch.Tensor,
                               mask: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Infectiousness-weighted prevalence within each group among ``mask``."""
    g = group.to(torch.int64)
    num = torch.zeros(n_groups, dtype=torch.float64)
    den = torch.zeros(n_groups, dtype=torch.float64)
    num.index_add_(0, g[mask], weight[mask])
    den.index_add_(0, g[mask], torch.ones(int(mask.sum()), dtype=torch.float64))
    return torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))


def group_hazards(c: Cohort, p: Params, active: torch.Tensor,
                  frozen: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """Annual sexual and injection hazards per exposure group."""
    exposure = torch.tensor(p.vector("exposure_rate_sex_g", N_EXPOSURE_GROUPS), dtype=torch.float64)

    counts = torch.zeros(N_EXPOSURE_GROUPS, dtype=torch.float64)
    counts.index_add_(0, c.exposure_group.to(torch.int64)[active],
                      torch.ones(int(active.sum()), dtype=torch.float64))
    mixing = MixingModel.build(p, counts * exposure).matrix

    if frozen is not None:
        prev_sex, prev_inj = frozen["sexual"], frozen["injection"]
    else:
        w_sex = infectiousness_sexual(c.hiv_stage, c.care_state, p)
        prev_sex = _group_weighted_prevalence(c.exposure_group, w_sex, active, N_EXPOSURE_GROUPS)
        sharers = active & c.shares_equipment
        w_inj = infectiousness_injection(c.hiv_stage, c.care_state, p)
        prev_inj = _group_weighted_prevalence(c.exposure_group, w_inj, sharers, N_EXPOSURE_GROUPS)

    lam_sex = p["lambda_sex_scale"] * exposure * (mixing @ prev_sex) + p["external_hazard_sex_annual"]
    lam_inj = p["lambda_inj_scale"] * (mixing @ prev_inj) + p["external_hazard_inj_annual"]
    return {"sexual": lam_sex, "injection": lam_inj,
            "prevalence_sexual": prev_sex, "prevalence_injection": prev_inj}


def baseline_prevalence(c: Cohort, p: Params, active: torch.Tensor) -> dict[str, torch.Tensor]:
    """Snapshot used by the direct-effects-only structural scenario."""
    h = group_hazards(c, p, active)
    return {"sexual": h["prevalence_sexual"], "injection": h["prevalence_injection"]}


def step_infections(c: Cohort, seed: int, step: int, dt: float, p: Params,
                    active: torch.Tensor, e_sex: torch.Tensor, e_inj: torch.Tensor,
                    frozen: dict[str, torch.Tensor] | None = None) -> dict[str, int]:
    """Draw new infections for susceptible people and move them to acute.

    ``e_sex`` and ``e_inj`` are current protection in [0, 1], already reflecting
    product, dosing state and waning.
    """
    lam = group_hazards(c, p, active, frozen)
    g = c.exposure_group.to(torch.int64)
    lam_sex = lam["sexual"][g]
    lam_inj = torch.where(c.shares_equipment, lam["injection"][g], torch.zeros_like(lam_sex))

    hazard = lam_sex * (1.0 - e_sex) + lam_inj * (1.0 - e_inj)
    p_inf = 1.0 - torch.exp(-hazard.clamp_min(0.0) * dt)

    susceptible = active & (c.hiv_stage == int(HivStage.SUSCEPTIBLE))
    infected = susceptible & (uniform(seed, Stream.INFECTION, step, c.uid) < p_inf)

    c.hiv_stage = torch.where(infected, torch.tensor(int(HivStage.ACUTE), dtype=torch.int8),
                              c.hiv_stage)
    c.time_since_infection = torch.where(infected, torch.zeros_like(c.time_since_infection),
                                         c.time_since_infection)
    c.infected_during_run = c.infected_during_run | infected
    # Acquiring HIV ends PrEP. Prevention is not a substitute for treatment.
    c.on_program = c.on_program & ~infected

    sex_share = torch.where(hazard > 0, lam_sex * (1 - e_sex) / hazard.clamp_min(1e-12),
                            torch.zeros_like(hazard))
    return {
        "infections": int(infected.sum()),
        "infections_sexual_expected": float((infected.to(torch.float64) * sex_share).sum()),
    }
