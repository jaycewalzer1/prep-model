"""Stage 1: the transparent analytic check.

One year of prevention expenditure against the lifetime consequences of the
infections avoided during that year. It exists to catch order-of-magnitude
errors before any transmission dynamics are built, and it is the only place in
this package where a single lifetime cost figure is used at all.

What it deliberately omits: secondary transmission, infection timing within the
year, adverse events, changing eligibility, heterogeneous risk, and the
recurring cost of a multi-year programme. It does not say that savings arrive
in year one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class BreakEven:
    annual_risk: float
    effective_protection: float
    incremental_lifetime_cost: float
    participants: int
    infections_averted: float
    lifetime_cost_offset: float
    break_even_cost_per_participant_year: float
    approximation: str = (
        "small-risk approximation: infections averted = participants * risk * protection"
    )

    def as_dict(self) -> dict:
        return asdict(self)


def break_even(
    annual_risk: float,
    effective_protection: float,
    incremental_lifetime_cost: float,
    participants: int = 1000,
    hiv_attributable_housing_saving: float = 0.0,
) -> BreakEven:
    """Break-even all-in programme cost per enrolled person-year.

    ``hiv_attributable_housing_saving`` defaults to zero on purpose. Adding an
    assumed shelter saving changes the answer by construction, so it has to be
    passed in deliberately and sourced.
    """
    if not 0.0 <= annual_risk <= 1.0:
        raise ValueError(f"annual_risk {annual_risk} is not a probability")
    if not 0.0 <= effective_protection <= 1.0:
        raise ValueError(f"effective_protection {effective_protection} is not a proportion")
    averted = participants * annual_risk * effective_protection
    offset_per_infection = incremental_lifetime_cost + hiv_attributable_housing_saving
    offset = averted * offset_per_infection
    per_person = annual_risk * effective_protection * offset_per_infection
    return BreakEven(
        annual_risk=annual_risk,
        effective_protection=effective_protection,
        incremental_lifetime_cost=incremental_lifetime_cost,
        participants=participants,
        infections_averted=averted,
        lifetime_cost_offset=offset,
        break_even_cost_per_participant_year=per_person,
    )


def break_even_table(
    risks: list[float],
    effective_protection: float,
    incremental_lifetime_cost: float,
    participants: int = 1000,
) -> list[BreakEven]:
    return [break_even(r, effective_protection, incremental_lifetime_cost, participants)
            for r in risks]


def exact_infections_averted(annual_risk: float, effective_protection: float,
                             participants: int = 1000) -> float:
    """The same quantity without the small-risk approximation.

    Uses constant hazards over the year: ``h = -ln(1 - risk)``, protected hazard
    ``h * (1 - protection)``. Reported beside the approximation so the size of
    the approximation error is visible rather than assumed negligible.
    """
    import math

    if annual_risk >= 1.0:
        raise ValueError("annual_risk must be below 1 for a hazard conversion")
    h = -math.log(1.0 - annual_risk)
    p_unprotected = 1.0 - math.exp(-h)
    p_protected = 1.0 - math.exp(-h * (1.0 - effective_protection))
    return participants * (p_unprotected - p_protected)
