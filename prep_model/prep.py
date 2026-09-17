"""Regimen exposure: what protection a person actually has this week.

Protection is a function of the product and of how long it has been since the
last administration. There is no efficacy cliff on the due date. Coverage is
full through the dosing interval and then declines linearly to zero over a
configured tail, and that tail is an uncertainty scenario, not a validated
pharmacokinetic curve.

Route specificity is the other thing this module enforces. Injectable efficacy
against equipment-sharing acquisition is zero in the primary analysis. That is
a conservative structural choice reflecting that CDC identifies prevention in
people who inject drugs as needing further research; it is not a claim that the
biological effect is known to be zero. People who share equipment still receive
the full modelled benefit against sexual acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .params import Params
from .population import Cohort
from .states import Product


@dataclass(frozen=True)
class Regimen:
    product: Product
    first_interval_weeks: float   # weeks from dose 1 to dose 2
    interval_weeks: float         # maintenance interval
    tail_weeks: float             # weeks of declining protection past the due date
    efficacy_sexual: float
    efficacy_injection: float
    cost_per_dose: float

    def due_interval(self, doses_received: torch.Tensor) -> torch.Tensor:
        """Weeks until the next dose is due, given how many doses are in."""
        return torch.where(doses_received == 1,
                           torch.full_like(doses_received, self.first_interval_weeks,
                                           dtype=torch.float64),
                           torch.full_like(doses_received, self.interval_weeks,
                                           dtype=torch.float64))


def regimen_for(product_name: str, p: Params) -> Regimen | None:
    if product_name == "lenacapavir":
        return Regimen(
            product=Product.LENACAPAVIR,
            first_interval_weeks=p["len_dose_interval_weeks"],
            interval_weeks=p["len_dose_interval_weeks"],
            tail_weeks=p["len_tail_weeks"],
            efficacy_sexual=p["len_efficacy_sexual"],
            efficacy_injection=p["injectable_efficacy_injection_route"],
            cost_per_dose=p["cost_len_per_dose"],
        )
    if product_name == "cabotegravir":
        return Regimen(
            product=Product.CABOTEGRAVIR,
            first_interval_weeks=p["cab_first_interval_weeks"],
            interval_weeks=p["cab_dose_interval_weeks"],
            tail_weeks=p["cab_tail_weeks"],
            efficacy_sexual=p["cab_efficacy_sexual"],
            efficacy_injection=p["injectable_efficacy_injection_route"],
            cost_per_dose=p["cost_cab_per_dose"],
        )
    if product_name == "none":
        return None
    raise ValueError(f"unknown product {product_name!r}")


def coverage(weeks_since_dose: torch.Tensor, due_weeks: torch.Tensor,
             tail_weeks: float) -> torch.Tensor:
    """Fraction of full protection retained, in [0, 1]."""
    over = (weeks_since_dose - due_weeks).clamp_min(0.0)
    if tail_weeks <= 0:
        return (over <= 0).to(torch.float64)
    return (1.0 - over / tail_weeks).clamp(0.0, 1.0)


def protection(c: Cohort, regimen: Regimen | None, p: Params) -> tuple[torch.Tensor, torch.Tensor]:
    """Current sexual and injection-route protection for everyone.

    Oral PrEP protection sits alongside the injectable, so a person who has
    lapsed off an injection falls back to whatever they are still taking rather
    than dropping to zero by construction.
    """
    zeros = torch.zeros(c.age.shape, dtype=torch.float64)
    on_oral = c.product == int(Product.ORAL)
    e_sex = torch.where(on_oral, torch.full_like(zeros, p["oral_efficacy_sexual"]), zeros)
    e_inj = torch.where(on_oral, torch.full_like(zeros, p["oral_efficacy_injection_route"]), zeros)

    if regimen is None:
        return e_sex, e_inj

    on_inj = c.on_program & (c.product == int(regimen.product)) & (c.doses_received > 0)
    cov = coverage(c.weeks_since_dose, regimen.due_interval(c.doses_received), regimen.tail_weeks)
    inj_sex = cov * regimen.efficacy_sexual
    inj_inj = cov * regimen.efficacy_injection
    # A person is not protected twice over; take the better of the two.
    e_sex = torch.where(on_inj, torch.maximum(e_sex, inj_sex), e_sex)
    e_inj = torch.where(on_inj, torch.maximum(e_inj, inj_inj), e_inj)
    return e_sex.clamp(0.0, 1.0), e_inj.clamp(0.0, 1.0)


def is_due(c: Cohort, regimen: Regimen) -> torch.Tensor:
    return c.on_program & (c.doses_received > 0) & \
        (c.weeks_since_dose >= regimen.due_interval(c.doses_received))


def past_tail(c: Cohort, regimen: Regimen) -> torch.Tensor:
    """Protection has reached zero; the person counts as discontinued."""
    return c.on_program & (c.doses_received > 0) & \
        (c.weeks_since_dose > regimen.due_interval(c.doses_received) + regimen.tail_weeks)
