"""The programme as a delivery system.

reachable -> offered -> accepts -> tested -> eligible -> first administration ->
subsequent administration -> retained or re-engaged.

Every one of those arrows can fail, and failing costs money. Contacts that do
not end in an injection are charged. Acceptance is not initiation. Capacity for
staff, slots and outreach routes is finite, and when it binds, people who would
have started do not start.

Two choices worth naming. Injection visits are served before new outreach when
capacity is short, which is a policy choice, not a neutral default. And reach
can be made to depend on exposure through ``reach_exposure_gradient``: at 1.0
reach is unrelated to exposure, and below 1.0 the people at greatest risk are
the hardest to keep, which is the scenario in which average coverage most
overstates real protection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import Arm
from .economics import Ledger
from .params import Params
from .population import Cohort
from .prep import Regimen, is_due, past_tail
from .rng import Stream, uniform
from .states import CareState, HivStage, Housing, Product


@dataclass
class DeliveryCounters:
    contacts: int = 0
    tests: int = 0
    offers: int = 0
    acceptances: int = 0
    initiations: int = 0
    doses: int = 0
    missed_visits: int = 0
    reengagements: int = 0
    discontinuations: int = 0
    program_diagnoses: int = 0
    linked_to_art: int = 0
    switchers_from_oral: int = 0
    capacity_blocked_initiations: int = 0
    capacity_blocked_contacts: int = 0
    steps_capacity_bound: int = 0

    def add(self, other: "DeliveryCounters") -> None:
        for k, v in other.__dict__.items():
            setattr(self, k, getattr(self, k) + v)

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class DeliverySettings:
    reach_rate: float
    reach_unsheltered_multiplier: float
    accept_prob: float
    initiate_given_accept: float
    attendance_prob: float
    reengage_prob_weekly: float
    contact_cost: float
    exposure_gradient: float
    capacity_initiations: float   # per 1000 eligible per week
    capacity_contacts: float      # per 1000 eligible per week


def settings_for(arm: Arm, p: Params) -> DeliverySettings | None:
    if arm.delivery == "none":
        return None
    if arm.delivery == "clinic":
        return DeliverySettings(
            reach_rate=p["reach_rate_annual_clinic"],
            reach_unsheltered_multiplier=p["reach_multiplier_unsheltered"],
            accept_prob=p["accept_prob_clinic"],
            initiate_given_accept=p["initiate_given_accept_clinic"],
            attendance_prob=p["attendance_prob_clinic"],
            reengage_prob_weekly=p["reengage_prob_weekly_after_lapse"] * 0.5,
            contact_cost=p["cost_contact_clinic"],
            exposure_gradient=p["reach_exposure_gradient"],
            capacity_initiations=p["capacity_initiations_per_1000_per_week"],
            capacity_contacts=p["capacity_contacts_per_1000_per_week"],
        )
    return DeliverySettings(
        reach_rate=p["reach_rate_annual_lowbarrier"],
        reach_unsheltered_multiplier=p["reach_multiplier_unsheltered_lowbarrier"],
        accept_prob=p["accept_prob_lowbarrier"],
        initiate_given_accept=p["initiate_given_accept_lowbarrier"],
        attendance_prob=p["attendance_prob_lowbarrier"],
        reengage_prob_weekly=p["reengage_prob_weekly_after_lapse"],
        contact_cost=p["cost_contact_lowbarrier"],
        exposure_gradient=p["reach_exposure_gradient"],
        capacity_initiations=p["capacity_initiations_per_1000_per_week"],
        capacity_contacts=p["capacity_contacts_per_1000_per_week"],
    )


def _cap(mask: torch.Tensor, priority: torch.Tensor, limit: int) -> tuple[torch.Tensor, int]:
    """Keep at most ``limit`` of ``mask``, preferring the lowest ``priority``.

    Ties are broken by the draw that produced the priority, never by row order.
    """
    n = int(mask.sum())
    if limit >= n:
        return mask, 0
    if limit <= 0:
        return torch.zeros_like(mask), n
    idx = torch.nonzero(mask, as_tuple=True)[0]
    order = torch.argsort(priority[idx])
    keep = torch.zeros_like(mask)
    keep[idx[order[:limit]]] = True
    return keep, n - limit


def _charge_visit(ledger: Ledger, s: DeliverySettings, p: Params, step: int, n: int) -> None:
    if n == 0:
        return
    ledger.accrue("program_delivery", s.contact_cost * n, step)
    ledger.accrue("patient_time", p["cost_patient_time_per_visit"] * n, step)


def step_delivery(c: Cohort, arm: Arm, s: DeliverySettings, regimen: Regimen | None,
                  ledger: Ledger, seed: int, step: int, dt: float, p: Params,
                  active: torch.Tensor) -> DeliveryCounters:
    """One week of offers, injections, lapses and returns."""
    k = DeliveryCounters()
    eligible_pool = int(active.sum())
    contact_budget = int(s.capacity_contacts * eligible_pool / 1000.0)
    init_budget = int(s.capacity_initiations * eligible_pool / 1000.0)

    # ---- 1. Scheduled injection visits come first when capacity is short. ----
    if regimen is not None and arm.injectable_uptake:
        due = active & is_due(c, regimen)
        u_attend = uniform(seed, Stream.PREP_ATTEND, step, c.uid)
        u_reeng = uniform(seed, Stream.PREP_REENGAGE, step, c.uid)

        # Attendance is modulated by the person's latent engagement, so retention
        # correlates with the same trait that drives reach and care engagement.
        attend_p = torch.full(c.age.shape, s.attendance_prob, dtype=torch.float64)
        attend_p = (attend_p * (0.6 + 0.8 * c.engagement)).clamp(0.0, 1.0)

        on_time = due & ~c.lapsed & (u_attend < attend_p)
        missed = due & ~c.lapsed & ~on_time
        returning = due & c.lapsed & (u_reeng < s.reengage_prob_weekly)
        attending = on_time | returning

        attending, blocked = _cap(attending, u_attend, contact_budget)
        k.capacity_blocked_contacts += blocked
        contact_budget -= int(attending.sum())

        n_att = int(attending.sum())
        if n_att:
            _charge_visit(ledger, s, p, step, n_att)
            ledger.accrue("program_testing", p["cost_hiv_test"] * n_att, step)
            ledger.accrue("program_drug", regimen.cost_per_dose * n_att, step)
            ledger.discounted_doses += n_att * ledger.factor(step)
            ledger.accrue("program_delivery", p["cost_adverse_event_annual"] * dt * n_att, step)
            k.tests += n_att
            k.doses += n_att
            c.doses_received = torch.where(attending, c.doses_received + 1, c.doses_received)
            c.weeks_since_dose = torch.where(attending, torch.zeros_like(c.weeks_since_dose),
                                             c.weeks_since_dose)
            c.lapsed = c.lapsed & ~attending
        k.missed_visits += int(missed.sum())
        k.reengagements += int(returning.sum())
        c.lapsed = c.lapsed | missed

        # Discontinuation: protection has run out, or an adverse event.
        u_ae = uniform(seed, Stream.PREP_ADVERSE_EVENT, step, c.uid)
        ae_stop = active & c.on_program & \
            (u_ae < 1.0 - torch.exp(torch.tensor(-p["ae_discontinuation_rate_annual"] * dt,
                                                 dtype=torch.float64)))
        stopped = (active & past_tail(c, regimen)) | ae_stop
        if bool(stopped.any()):
            k.discontinuations += int(stopped.sum())
            c.on_program = c.on_program & ~stopped
            c.lapsed = c.lapsed & ~stopped
            to_oral = stopped & (uniform(seed, Stream.PREP_POST_STOP_ORAL, step, c.uid)
                                 < p["oral_after_stopping_prob"])
            c.product = torch.where(stopped,
                                    torch.tensor(int(Product.NONE), dtype=torch.int8), c.product)
            c.product = torch.where(to_oral,
                                    torch.tensor(int(Product.ORAL), dtype=torch.int8), c.product)

    # ---- 2. Outreach contacts with whatever capacity is left. ----
    reachable = active & ~c.on_program & ~c.refused
    rate = torch.full(c.age.shape, s.reach_rate, dtype=torch.float64)
    rate = torch.where(c.housing == int(Housing.UNSHELTERED),
                       rate * s.reach_unsheltered_multiplier, rate)
    rate = rate * (s.exposure_gradient ** c.exposure_group.to(torch.float64))
    rate = rate * (0.6 + 0.8 * c.engagement)
    p_reach = 1.0 - torch.exp(-rate.clamp_min(0.0) * dt)
    u_reach = uniform(seed, Stream.PREP_REACH, step, c.uid)
    contacted = reachable & (u_reach < p_reach)
    contacted, blocked = _cap(contacted, u_reach, contact_budget)
    k.capacity_blocked_contacts += blocked
    if blocked:
        k.steps_capacity_bound += 1

    n_contacts = int(contacted.sum())
    k.contacts += n_contacts
    _charge_visit(ledger, s, p, step, n_contacts)

    # ---- 3. HIV screening at the contact. ----
    known_positive = c.care_state >= int(CareState.DIAGNOSED_NO_ART)
    tested = contacted & ~known_positive
    n_tests = int(tested.sum())
    k.tests += n_tests
    ledger.accrue("program_testing", p["cost_hiv_test"] * n_tests, step)

    u_test = uniform(seed, Stream.PROGRAM_TEST, step, c.uid)
    detectable = (c.hiv_stage == int(HivStage.CHRONIC)) | (c.hiv_stage == int(HivStage.ADVANCED)) | \
        ((c.hiv_stage == int(HivStage.ACUTE)) & (u_test < p["hiv_test_sensitivity_acute"]))
    found = tested & (c.hiv_stage != int(HivStage.SUSCEPTIBLE)) & detectable
    k.program_diagnoses += int(found.sum())
    c.care_state = torch.where(found,
                               torch.tensor(int(CareState.DIAGNOSED_NO_ART), dtype=torch.int8),
                               c.care_state)
    linked = found & (uniform(seed, Stream.PROGRAM_LINKAGE, step, c.uid)
                      < p["linkage_prob_program_diagnosis"])
    k.linked_to_art += int(linked.sum())
    c.care_state = torch.where(linked,
                               torch.tensor(int(CareState.ART_UNSUPPRESSED), dtype=torch.int8),
                               c.care_state)

    # ---- 4. Offer, acceptance and first administration. ----
    # Arm E stops here: identical outreach, testing and linkage, no injectable uptake.
    offered = contacted & ~found & ~known_positive
    k.offers += int(offered.sum())
    c.ever_offered = c.ever_offered | offered

    if regimen is None or not arm.injectable_uptake:
        return k

    u_accept = uniform(seed, Stream.PREP_ACCEPT, step, c.uid)
    accepted = offered & (u_accept < s.accept_prob)
    k.acceptances += int(accepted.sum())
    declined = offered & ~accepted
    c.refused = c.refused | (declined & (uniform(seed, Stream.PREP_REFUSAL_PERMANENT, step, c.uid)
                                         < p["refusal_permanent_prob"]))

    u_init = uniform(seed, Stream.PREP_INITIATE, step, c.uid)
    starting = accepted & (u_init < s.initiate_given_accept)
    starting, blocked = _cap(starting, u_init, init_budget)
    k.capacity_blocked_initiations += blocked
    if blocked:
        k.steps_capacity_bound += 1

    n_start = int(starting.sum())
    if n_start:
        k.initiations += n_start
        k.doses += n_start
        k.switchers_from_oral += int((starting & (c.product == int(Product.ORAL))).sum())
        ledger.accrue("program_drug", regimen.cost_per_dose * n_start, step)
        ledger.discounted_doses += n_start * ledger.factor(step)
        c.switched_from_oral = c.switched_from_oral | (starting & (c.product == int(Product.ORAL)))
        c.on_program = c.on_program | starting
        c.ever_initiated = c.ever_initiated | starting
        c.product = torch.where(starting,
                                torch.tensor(int(regimen.product), dtype=torch.int8), c.product)
        c.doses_received = torch.where(starting, torch.ones_like(c.doses_received),
                                       c.doses_received)
        c.weeks_since_dose = torch.where(starting, torch.zeros_like(c.weeks_since_dose),
                                         c.weeks_since_dose)
        c.lapsed = c.lapsed & ~starting
    return k


def accrue_background_prevention(ledger: Ledger, c: Cohort, active: torch.Tensor,
                                 p: Params, step: int, dt: float) -> None:
    """Existing daily oral PrEP is part of usual care and is costed in every arm."""
    on_oral = active & (c.product == int(Product.ORAL))
    ledger.accrue("program_drug", p["cost_oral_prep_annual"] * dt * float(on_oral.sum()), step)


def accrue_fixed_program_cost(ledger: Ledger, arm: Arm, n_eligible: int, p: Params,
                              step: int, dt: float) -> None:
    """Supervision, vehicles, cold chain and data systems do not scale per injection."""
    if arm.delivery == "none":
        return
    ledger.accrue("program_delivery",
                  p["cost_program_fixed_annual_per_1000"] * n_eligible / 1000.0 * dt, step)
