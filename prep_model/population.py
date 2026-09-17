"""Synthetic cohort construction.

Two things matter here beyond building plausible people.

First, identity: every person has a stable ``uid``, and every attribute is a
pure function of that uid. Arms therefore start from a byte-identical
population and keep drawing the same numbers for the same person.

Second, inflow: entries into the service population are scheduled
deterministically from the configured inflow rate, not generated in response to
deaths. If entries backfilled deaths, an arm that prevented deaths would admit
fewer people, the uid sequence would diverge, and paired comparison would
quietly stop being paired. Population size is therefore allowed to drift, which
is also the more honest reading of an exogenous service inflow.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .params import Params
from .rng import Stream, categorical, normal, uniform
from .states import CareState, HivStage, Housing, N_EXPOSURE_GROUPS, Product


@dataclass
class Cohort:
    """Person-level state as separate one-dimensional tensors."""

    uid: torch.Tensor            # int64, stable identity
    entry_step: torch.Tensor     # int64, step at which the person enters
    age: torch.Tensor            # float64, years
    housing: torch.Tensor        # int8, Housing
    exposure_group: torch.Tensor  # int8
    shares_equipment: torch.Tensor  # bool
    engagement: torch.Tensor     # float64 in [0, 1), latent service-engagement propensity

    alive: torch.Tensor          # bool
    resident: torch.Tensor       # bool, still in the modelled catchment
    hiv_stage: torch.Tensor      # int8, HivStage
    care_state: torch.Tensor     # int8, CareState
    time_since_infection: torch.Tensor  # float64, years

    product: torch.Tensor        # int8, Product
    on_program: torch.Tensor     # bool, currently enrolled in the injectable programme
    weeks_since_dose: torch.Tensor  # float64
    doses_received: torch.Tensor    # int64
    ever_offered: torch.Tensor      # bool
    ever_initiated: torch.Tensor    # bool
    switched_from_oral: torch.Tensor  # bool
    lapsed: torch.Tensor            # bool, missed a due visit and not yet re-engaged
    refused: torch.Tensor           # bool, declined and expected to decline again
    person_years_on_program: torch.Tensor  # float64

    hiv_deaths: torch.Tensor     # bool, died with HIV as the modelled cause
    infected_during_run: torch.Tensor  # bool
    entered_flag: torch.Tensor   # bool, computed each step

    def clone(self) -> "Cohort":
        return Cohort(**{f.name: getattr(self, f.name).clone() for f in fields(self)})

    @property
    def n(self) -> int:
        return int(self.uid.shape[0])

    def active(self, step: int) -> torch.Tensor:
        """People who have entered, are alive, and are still in the catchment."""
        return (self.entry_step <= step) & self.alive & self.resident


def entry_schedule(n_baseline: int, inflow_rate_per_year: float, n_steps: int,
                   dt: float) -> torch.Tensor:
    """Entries per step, deterministic and identical across arms.

    Cumulative expected entries are floored, so the schedule is exact arithmetic
    with no draw at all.
    """
    steps = torch.arange(1, n_steps + 1, dtype=torch.float64)
    cum = torch.floor(n_baseline * inflow_rate_per_year * dt * steps)
    prev = torch.cat([torch.zeros(1, dtype=torch.float64), cum[:-1]])
    return (cum - prev).to(torch.int64)


def _draw_attributes(seed: int, uid: torch.Tensor, p: Params) -> dict[str, torch.Tensor]:
    n = uid.shape[0]

    age = p["age_mean"] + p["age_sd"] * normal(seed, Stream.ATTR_AGE, 0, uid)
    age = age.clamp(p["age_min"], p["age_max"])

    housed_share = max(0.0, 1.0 - p["frac_unsheltered"] - p["frac_sheltered"])
    total = p["frac_unsheltered"] + p["frac_sheltered"] + housed_share
    housing_probs = torch.tensor(
        [p["frac_unsheltered"] / total, p["frac_sheltered"] / total, housed_share / total],
        dtype=torch.float64,
    ).expand(n, 3)
    housing = categorical(uniform(seed, Stream.ATTR_HOUSING, 0, uid), housing_probs)

    shares = torch.tensor(p.vector("exposure_share_g", N_EXPOSURE_GROUPS), dtype=torch.float64)
    shares = shares / shares.sum()
    exposure = categorical(uniform(seed, Stream.ATTR_EXPOSURE, 0, uid),
                           shares.expand(n, N_EXPOSURE_GROUPS))

    inject_by_group = torch.tensor(p.vector("frac_injects_g", N_EXPOSURE_GROUPS),
                                   dtype=torch.float64)
    shares_equipment = uniform(seed, Stream.ATTR_INJECTS, 0, uid) < inject_by_group[exposure]

    engagement = uniform(seed, Stream.ATTR_ENGAGEMENT, 0, uid)

    infected = uniform(seed, Stream.ATTR_HIV, 0, uid) < p["baseline_hiv_prevalence"]

    # Care cascade among prevalent infections.
    u_care = uniform(seed, Stream.ATTR_CARE, 0, uid)
    undx = p["baseline_undiagnosed_fraction"]
    on_art = (1 - undx) * p["baseline_art_among_diagnosed"]
    suppressed = on_art * p["baseline_suppressed_among_art"]
    care = torch.full((n,), int(CareState.UNDIAGNOSED), dtype=torch.int8)
    care = torch.where(u_care >= undx, torch.tensor(int(CareState.DIAGNOSED_NO_ART), dtype=torch.int8), care)
    care = torch.where(u_care >= 1 - on_art, torch.tensor(int(CareState.ART_UNSUPPRESSED), dtype=torch.int8), care)
    care = torch.where(u_care >= 1 - suppressed, torch.tensor(int(CareState.ART_SUPPRESSED), dtype=torch.int8), care)
    care = torch.where(infected, care, torch.tensor(int(CareState.UNDIAGNOSED), dtype=torch.int8))

    # Stage split. Untreated infection carries more advanced disease.
    u_stage = uniform(seed, Stream.ATTR_STAGE, 0, uid)
    advanced_prob = torch.where(
        care <= int(CareState.DIAGNOSED_NO_ART),
        torch.tensor(0.12, dtype=torch.float64),
        torch.tensor(0.05, dtype=torch.float64),
    )
    stage = torch.where(
        infected,
        torch.where(u_stage < advanced_prob,
                    torch.tensor(int(HivStage.ADVANCED), dtype=torch.int8),
                    torch.tensor(int(HivStage.CHRONIC), dtype=torch.int8)),
        torch.tensor(int(HivStage.SUSCEPTIBLE), dtype=torch.int8),
    )

    tsi = torch.where(
        infected,
        uniform(seed, Stream.ATTR_TIME_SINCE_INFECTION, 0, uid) * 8.0,
        torch.zeros(n, dtype=torch.float64),
    )

    on_oral = (~infected) & (uniform(seed, Stream.ATTR_ORAL_PREP, 0, uid) < p["baseline_oral_prep_use"])
    product = torch.where(on_oral,
                          torch.tensor(int(Product.ORAL), dtype=torch.int8),
                          torch.tensor(int(Product.NONE), dtype=torch.int8))

    return {
        "age": age,
        "housing": housing.to(torch.int8),
        "exposure_group": exposure.to(torch.int8),
        "shares_equipment": shares_equipment,
        "engagement": engagement,
        "hiv_stage": stage,
        "care_state": care,
        "time_since_infection": tsi,
        "product": product,
    }


def build_cohort(seed: int, n_baseline: int, n_steps: int, dt: float, p: Params) -> Cohort:
    """Allocate everyone who will ever appear, baseline members and entrants alike.

    Entrants exist in the tensors from the start but are inert until their
    ``entry_step``. Pre-allocating is what makes an entrant's attributes a pure
    function of the seed rather than of what happened before they arrived.
    """
    schedule = entry_schedule(n_baseline, p["inflow_rate_per_year"], n_steps, dt)
    n_total = n_baseline + int(schedule.sum().item())
    uid = torch.arange(n_total, dtype=torch.int64)
    entry_step = torch.zeros(n_total, dtype=torch.int64)
    cursor = n_baseline
    for step, count in enumerate(schedule.tolist()):
        if count:
            entry_step[cursor:cursor + count] = step + 1
            cursor += count

    attrs = _draw_attributes(seed, uid, p)
    false = torch.zeros(n_total, dtype=torch.bool)
    return Cohort(
        uid=uid,
        entry_step=entry_step,
        alive=torch.ones(n_total, dtype=torch.bool),
        resident=torch.ones(n_total, dtype=torch.bool),
        on_program=false.clone(),
        weeks_since_dose=torch.zeros(n_total, dtype=torch.float64),
        doses_received=torch.zeros(n_total, dtype=torch.int64),
        ever_offered=false.clone(),
        ever_initiated=false.clone(),
        switched_from_oral=false.clone(),
        lapsed=false.clone(),
        refused=false.clone(),
        person_years_on_program=torch.zeros(n_total, dtype=torch.float64),
        hiv_deaths=false.clone(),
        infected_during_run=false.clone(),
        entered_flag=false.clone(),
        **attrs,
    )
