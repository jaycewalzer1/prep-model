"""Housing transitions.

Three states with movement in both directions. Leaving homelessness is not
leaving the model: a housed participant is still alive, may still be exposed,
and keeps accruing costs and outcomes. Out-migration is handled separately in
``natural_history.step_exits`` so that "moved away", "became housed" and "died"
are never confused with one another.

In the primary analysis PrEP has no effect on housing transitions. That is an
assumption, stated here rather than buried: preventing an infection does not
mean a person stops needing shelter.
"""

from __future__ import annotations

import torch

from .params import Params
from .population import Cohort
from .rng import Stream, uniform
from .states import Housing

_RATE_KEYS = {
    Housing.UNSHELTERED: [(Housing.SHELTERED, "housing_rate_u_to_s"),
                          (Housing.HOUSED, "housing_rate_u_to_h")],
    Housing.SHELTERED: [(Housing.UNSHELTERED, "housing_rate_s_to_u"),
                        (Housing.HOUSED, "housing_rate_s_to_h")],
    Housing.HOUSED: [(Housing.UNSHELTERED, "housing_rate_h_to_u"),
                     (Housing.SHELTERED, "housing_rate_h_to_s")],
}


def step_housing(c: Cohort, seed: int, step: int, dt: float, p: Params,
                 active: torch.Tensor) -> dict[str, int]:
    """One coherent multinomial move per person per step."""
    u_move = uniform(seed, Stream.HOUSING_MOVE, step, c.uid)
    u_dest = uniform(seed, Stream.HOUSING_DEST, step, c.uid)
    new_housing = c.housing.clone()
    moves = 0
    for origin, destinations in _RATE_KEYS.items():
        here = active & (c.housing == int(origin))
        if not bool(here.any()):
            continue
        rates = [p[key] for _, key in destinations]
        total = sum(rates)
        p_move = 1.0 - torch.exp(torch.tensor(-total * dt, dtype=torch.float64))
        moving = here & (u_move < p_move)
        first_share = rates[0] / total if total > 0 else 0.0
        to_first = moving & (u_dest < first_share)
        to_second = moving & ~to_first
        new_housing = torch.where(to_first,
                                  torch.tensor(int(destinations[0][0]), dtype=torch.int8),
                                  new_housing)
        new_housing = torch.where(to_second,
                                  torch.tensor(int(destinations[1][0]), dtype=torch.int8),
                                  new_housing)
        moves += int(moving.sum())
    c.housing = new_housing
    return {"housing_moves": moves}


def housing_distribution(c: Cohort, active: torch.Tensor) -> dict[str, float]:
    n = int(active.sum())
    if n == 0:
        return {h.name.lower(): 0.0 for h in Housing}
    return {h.name.lower(): float((active & (c.housing == int(h))).sum()) / n for h in Housing}
