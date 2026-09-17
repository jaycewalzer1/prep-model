"""HIV natural history, the care cascade, and mortality.

Susceptible, acute, chronic, advanced. Diagnosis, ART, suppression, treatment
interruption, re-engagement. There is no transition back to HIV negative.
Advanced disease can improve to chronic on effective ART, which is immune
reconstitution, not clearing the infection: ``hiv_stage`` never returns to
SUSCEPTIBLE and ``time_since_infection`` never resets.
"""

from __future__ import annotations

import torch

from .params import Params
from .population import Cohort
from .rng import Stream, uniform
from .states import CareState, HivStage, Housing

_AGE_BANDS = [(0.0, 30.0, "bg_mortality_18_29"),
              (30.0, 45.0, "bg_mortality_30_44"),
              (45.0, 60.0, "bg_mortality_45_59"),
              (60.0, 200.0, "bg_mortality_60_plus")]


def background_mortality(age: torch.Tensor, housing: torch.Tensor, p: Params) -> torch.Tensor:
    """Annual non-HIV mortality hazard by age band, multiplied by housing state."""
    rate = torch.zeros_like(age)
    for lo, hi, key in _AGE_BANDS:
        rate = torch.where((age >= lo) & (age < hi), torch.full_like(age, p[key]), rate)
    mult = torch.ones_like(age)
    mult = torch.where(housing == int(Housing.UNSHELTERED),
                       torch.full_like(age, p["mortality_multiplier_unsheltered"]), mult)
    mult = torch.where(housing == int(Housing.SHELTERED),
                       torch.full_like(age, p["mortality_multiplier_sheltered"]), mult)
    return rate * mult


def hiv_mortality(stage: torch.Tensor, care: torch.Tensor, p: Params) -> torch.Tensor:
    """Annual HIV-attributable mortality hazard, zero for susceptible people."""
    rate = torch.zeros(stage.shape, dtype=torch.float64)
    rate = torch.where(stage == int(HivStage.CHRONIC),
                       torch.full_like(rate, p["chronic_mortality_excess"]), rate)
    rate = torch.where(stage == int(HivStage.ADVANCED),
                       torch.full_like(rate, p["advanced_mortality_rate"]), rate)
    on_art = care >= int(CareState.ART_UNSUPPRESSED)
    suppressed = care == int(CareState.ART_SUPPRESSED)
    mult = torch.ones_like(rate)
    mult = torch.where(on_art, torch.full_like(rate, p["art_mortality_multiplier"] * 2.0), mult)
    mult = torch.where(suppressed, torch.full_like(rate, p["art_mortality_multiplier"]), mult)
    return rate * mult


def infectiousness_sexual(stage: torch.Tensor, care: torch.Tensor, p: Params) -> torch.Tensor:
    """Relative sexual infectiousness. Durable suppression contributes zero."""
    w = torch.zeros(stage.shape, dtype=torch.float64)
    w = torch.where(stage == int(HivStage.ACUTE), torch.full_like(w, p["infectiousness_acute"]), w)
    w = torch.where(stage == int(HivStage.CHRONIC), torch.ones_like(w), w)
    w = torch.where(stage == int(HivStage.ADVANCED),
                    torch.full_like(w, p["infectiousness_advanced"]), w)
    return torch.where(care == int(CareState.ART_SUPPRESSED),
                       torch.full_like(w, p["infectiousness_suppressed_sexual"]) * w, w)


def infectiousness_injection(stage: torch.Tensor, care: torch.Tensor, p: Params) -> torch.Tensor:
    """Relative injection-route infectiousness.

    The zero-risk result for sustained suppression is a statement about sexual
    transmission. It is not extended here; suppression reduces this route by a
    configured factor and the factor is a stated assumption.
    """
    w = torch.zeros(stage.shape, dtype=torch.float64)
    w = torch.where(stage == int(HivStage.ACUTE), torch.full_like(w, p["infectiousness_acute"]), w)
    w = torch.where(stage == int(HivStage.CHRONIC), torch.ones_like(w), w)
    w = torch.where(stage == int(HivStage.ADVANCED),
                    torch.full_like(w, p["infectiousness_advanced"]), w)
    return torch.where(care == int(CareState.ART_SUPPRESSED),
                       w * p["infectiousness_suppressed_injection"], w)


def _fires(seed: int, stream: Stream, step: int, uid: torch.Tensor,
           rate: torch.Tensor, dt: float, mask: torch.Tensor) -> torch.Tensor:
    p_event = 1.0 - torch.exp(-rate.clamp_min(0.0) * dt)
    return mask & (uniform(seed, stream, step, uid) < p_event)


def step_disease(c: Cohort, seed: int, step: int, dt: float, p: Params,
                 active: torch.Tensor) -> dict[str, int]:
    """Advance disease stage and the care cascade by one step, in place."""
    with_hiv = active & (c.hiv_stage != int(HivStage.SUSCEPTIBLE))
    counts: dict[str, int] = {}

    # Acute to chronic. People infected during *this* step are excluded: they are
    # acute for at least the interval they were infected in. Without the guard a
    # share 1 - exp(-dt / acute_duration) of new infections would leave the acute
    # window without ever spending a step in it, and since acute infectiousness is
    # several times chronic, the epidemic would then depend on the step length.
    acute = with_hiv & (c.hiv_stage == int(HivStage.ACUTE)) & (c.time_since_infection > 0.0)
    rate = torch.full(c.age.shape, 1.0 / p["acute_duration_years"], dtype=torch.float64)
    to_chronic = _fires(seed, Stream.ACUTE_PROGRESSION, step, c.uid, rate, dt, acute)
    c.hiv_stage = torch.where(to_chronic,
                              torch.tensor(int(HivStage.CHRONIC), dtype=torch.int8), c.hiv_stage)

    # Chronic to advanced, only while not virally suppressed.
    chronic = with_hiv & (c.hiv_stage == int(HivStage.CHRONIC)) & ~to_chronic
    unsuppressed = c.care_state != int(CareState.ART_SUPPRESSED)
    rate = torch.full(c.age.shape, p["chronic_to_advanced_rate"], dtype=torch.float64)
    to_advanced = _fires(seed, Stream.DISEASE_PROGRESSION, step, c.uid, rate, dt,
                         chronic & unsuppressed)
    c.hiv_stage = torch.where(to_advanced,
                              torch.tensor(int(HivStage.ADVANCED), dtype=torch.int8), c.hiv_stage)
    counts["progressed_to_advanced"] = int(to_advanced.sum())

    # Advanced back to chronic on suppressive ART. The infection persists.
    advanced_sup = with_hiv & (c.hiv_stage == int(HivStage.ADVANCED)) & \
        (c.care_state == int(CareState.ART_SUPPRESSED))
    rate = torch.full(c.age.shape, p["art_regression_rate"], dtype=torch.float64)
    recovered = _fires(seed, Stream.ART_REGRESSION, step, c.uid, rate, dt, advanced_sup)
    c.hiv_stage = torch.where(recovered,
                              torch.tensor(int(HivStage.CHRONIC), dtype=torch.int8), c.hiv_stage)

    # Background diagnosis. Programme-driven testing is handled in delivery.py.
    undiagnosed = with_hiv & (c.care_state == int(CareState.UNDIAGNOSED))
    rate = torch.full(c.age.shape, p["testing_rate_annual"], dtype=torch.float64)
    rate = torch.where(c.hiv_stage == int(HivStage.ADVANCED),
                       rate * p["testing_rate_advanced_multiplier"], rate)
    diagnosed = _fires(seed, Stream.TESTING, step, c.uid, rate, dt, undiagnosed)
    c.care_state = torch.where(diagnosed,
                               torch.tensor(int(CareState.DIAGNOSED_NO_ART), dtype=torch.int8),
                               c.care_state)
    counts["background_diagnoses"] = int(diagnosed.sum())

    # ART initiation after diagnosis.
    off_art = with_hiv & (c.care_state == int(CareState.DIAGNOSED_NO_ART)) & ~diagnosed
    rate = torch.full(c.age.shape, p["art_init_rate_annual"], dtype=torch.float64)
    started = _fires(seed, Stream.ART_INIT, step, c.uid, rate, dt, off_art)
    c.care_state = torch.where(started,
                               torch.tensor(int(CareState.ART_UNSUPPRESSED), dtype=torch.int8),
                               c.care_state)
    counts["art_initiations"] = int(started.sum())

    # Suppression, loss of suppression, disengagement and return to care.
    unsup_on_art = with_hiv & (c.care_state == int(CareState.ART_UNSUPPRESSED)) & ~started
    rate = torch.full(c.age.shape, p["suppression_rate_on_art"], dtype=torch.float64)
    suppressed = _fires(seed, Stream.SUPPRESSION, step, c.uid, rate, dt, unsup_on_art)
    c.care_state = torch.where(suppressed,
                               torch.tensor(int(CareState.ART_SUPPRESSED), dtype=torch.int8),
                               c.care_state)

    sup = with_hiv & (c.care_state == int(CareState.ART_SUPPRESSED)) & ~suppressed
    rate = torch.full(c.age.shape, p["art_failure_rate"], dtype=torch.float64)
    failed = _fires(seed, Stream.ART_FAILURE, step, c.uid, rate, dt, sup)
    c.care_state = torch.where(failed,
                               torch.tensor(int(CareState.ART_UNSUPPRESSED), dtype=torch.int8),
                               c.care_state)

    on_art = with_hiv & (c.care_state >= int(CareState.ART_UNSUPPRESSED)) & ~suppressed & ~failed
    rate = torch.full(c.age.shape, p["disengage_rate_annual"], dtype=torch.float64)
    left = _fires(seed, Stream.DISENGAGE, step, c.uid, rate, dt, on_art)
    c.care_state = torch.where(left,
                               torch.tensor(int(CareState.DIAGNOSED_NO_ART), dtype=torch.int8),
                               c.care_state)
    counts["disengagements"] = int(left.sum())

    out_of_care = with_hiv & (c.care_state == int(CareState.DIAGNOSED_NO_ART)) & \
        ~left & ~started & ~diagnosed
    rate = torch.full(c.age.shape, p["reengage_rate_annual"], dtype=torch.float64)
    back = _fires(seed, Stream.REENGAGE, step, c.uid, rate, dt, out_of_care)
    c.care_state = torch.where(back,
                               torch.tensor(int(CareState.ART_UNSUPPRESSED), dtype=torch.int8),
                               c.care_state)

    c.time_since_infection = torch.where(with_hiv, c.time_since_infection + dt,
                                         c.time_since_infection)
    return counts


def step_exits(c: Cohort, seed: int, step: int, dt: float, p: Params,
               active: torch.Tensor) -> dict[str, int]:
    """Death and out-migration as coherent competing hazards.

    One draw decides whether the person leaves the modelled population at all;
    a second splits the cause in proportion to the two hazards. A person can
    therefore never both die and migrate in the same interval.
    """
    mort = background_mortality(c.age, c.housing, p) + hiv_mortality(c.hiv_stage, c.care_state, p)
    mig = torch.full(c.age.shape, p["migration_rate_per_year"], dtype=torch.float64)
    total = mort + mig
    p_exit = 1.0 - torch.exp(-total.clamp_min(0.0) * dt)
    exits = active & (uniform(seed, Stream.EXIT, step, c.uid) < p_exit)

    share_death = torch.where(total > 0, mort / total.clamp_min(1e-12), torch.zeros_like(total))
    died = exits & (uniform(seed, Stream.EXIT_CAUSE, step, c.uid) < share_death)
    migrated = exits & ~died

    hiv_share = torch.where(mort > 0, hiv_mortality(c.hiv_stage, c.care_state, p) / mort.clamp_min(1e-12),
                            torch.zeros_like(mort))
    hiv_death = died & (uniform(seed, Stream.DEATH_ATTRIBUTION, step, c.uid) < hiv_share)

    c.alive = c.alive & ~died
    c.resident = c.resident & ~migrated
    c.hiv_deaths = c.hiv_deaths | hiv_death
    c.on_program = c.on_program & ~(died | migrated)
    return {
        "deaths": int(died.sum()),
        "hiv_deaths": int(hiv_death.sum()),
        "migrations": int(migrated.sum()),
    }
