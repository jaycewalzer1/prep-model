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


@dataclass(frozen=True)
class InfectionBurden:
    """The discounted lifetime consequence of one infection in this population.

    Built by valuing two survival streams and differencing them, using the same
    state-specific costs and utilities the simulation uses, so this cannot drift
    away from the model it is meant to summarise.
    """

    discount_rate: float
    years_if_uninfected: float
    years_if_infected: float
    hiv_care: float
    non_hiv_care: float
    housing: float
    net_cost: float
    qalys_lost: float

    def value_at(self, wtp: float) -> float:
        """What averting one infection is worth: cost avoided plus health gained."""
        return self.net_cost + wtp * self.qalys_lost

    def as_dict(self) -> dict:
        return asdict(self)


def _annuity(life_expectancy: float, discount_rate: float) -> float:
    """Present value of one unit a year under a constant mortality hazard.

    The same exponential-survival annuity the terminal value uses: with hazard
    ``m = 1 / life expectancy`` and continuous discount rate ``rho``, a unit
    stream is worth ``1 / (rho + m)``.
    """
    import math

    rho = math.log(1.0 + discount_rate)
    m = 1.0 / life_expectancy
    return 1.0 / (rho + m)


def lifetime_burden_of_one_infection(p, discount_rate: float,
                                     suppressed: bool = True) -> InfectionBurden:
    """Discounted lifetime cost and QALY loss caused by a single infection.

    The comparison is a person who acquires HIV against the same person who does
    not. Both keep incurring non-HIV medical and housing costs for as long as
    they live, and the infected person lives less long, so those two categories
    come back *negative*: an infection saves the public sector money by shortening
    a life. That offset is real arithmetic and is reported rather than suppressed,
    because leaving it out would overstate the case for prevention.

    ``suppressed`` picks the optimistic cascade: diagnosed, on ART and virally
    suppressed for the rest of life. Setting it False uses the unsuppressed
    survival and cost, which is worse health and, because the person dies sooner,
    not necessarily a larger cost.
    """
    le_neg = p["life_expectancy_hiv_negative_at_45"]
    le_hiv = (p["life_expectancy_hiv_suppressed_at_45"] if suppressed
              else p["life_expectancy_hiv_unsuppressed_at_45"])
    a_neg = _annuity(le_neg, discount_rate)
    a_hiv = _annuity(le_hiv, discount_rate)

    hiv_annual = (p["cost_hiv_care_art_suppressed"] if suppressed
                  else p["cost_hiv_care_art_unsuppressed"])
    non_hiv = p["cost_non_hiv_care_annual"]
    # Housing at the population's own mix is arm-invariant; the HIV-attributable
    # increment is the only part a prevented infection changes, and it is zero
    # in the primary analysis unless a source is supplied.
    housing_annual = p["cost_hiv_attributable_housing_annual"]

    u_neg = p["utility_hiv_negative"]
    u_hiv = p["utility_hiv_art"]

    hiv_care = hiv_annual * a_hiv
    non_hiv_care = non_hiv * (a_hiv - a_neg)
    housing = housing_annual * a_hiv
    return InfectionBurden(
        discount_rate=discount_rate,
        years_if_uninfected=a_neg,
        years_if_infected=a_hiv,
        hiv_care=hiv_care,
        non_hiv_care=non_hiv_care,
        housing=housing,
        net_cost=hiv_care + non_hiv_care + housing,
        qalys_lost=u_neg * a_neg - u_hiv * a_hiv,
    )
