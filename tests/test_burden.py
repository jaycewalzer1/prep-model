"""The lifetime consequence of one infection is the number the whole case turns on.

A short-horizon run counts only the treatment an infected person receives before
the horizon, and then annuitises whatever state they were in when the clock
stopped. New infections are disproportionately undiagnosed at a short horizon,
and an undiagnosed person is costed at a fiftieth of the suppressed-on-ART rate,
so the short horizon undercounts the cost of an infection and undercounts it
worst in the arm that prevents infections.

These checks are about that quantity: that it is built from the same inputs the
simulation uses, that the uncomfortable part of it is not quietly dropped, and
that it is the right order of magnitude against the model's own parameters.
"""

from __future__ import annotations

import math

import pytest

from prep_model.analytic import _annuity, lifetime_burden_of_one_infection


@pytest.fixture(scope="module")
def burden(params):
    return lifetime_burden_of_one_infection(params, 0.03)


def test_the_annuity_is_the_one_the_terminal_value_uses(params):
    """If these two ever diverge, the summary stops describing the model."""
    le = params["life_expectancy_hiv_suppressed_at_45"]
    rho = math.log(1.03)
    assert _annuity(le, 0.03) == pytest.approx(1.0 / (rho + 1.0 / le), rel=1e-12)


def test_an_infection_costs_money_and_costs_health(burden):
    assert burden.net_cost > 0.0
    assert burden.qalys_lost > 0.0


def test_the_shorter_life_shows_up_as_a_saving_and_is_not_dropped(burden):
    """The part of the arithmetic nobody wants to write down.

    An infected person dies sooner and therefore stops consuming non-HIV care
    sooner. That is a genuine reduction in public expenditure and it offsets part
    of the treatment cost. A model that reported only the treatment cost would
    overstate the case for prevention, so the offset must be present, negative,
    and smaller than the treatment cost it offsets.
    """
    assert burden.non_hiv_care < 0.0
    assert abs(burden.non_hiv_care) < burden.hiv_care
    assert burden.net_cost == pytest.approx(
        burden.hiv_care + burden.non_hiv_care + burden.housing, rel=1e-12)


def test_the_lifetime_cost_is_dominated_by_treatment(burden, params):
    """Order of magnitude: lifetime ART at the model's own price and survival."""
    annual = params["cost_hiv_care_art_suppressed"]
    assert burden.hiv_care == pytest.approx(annual * burden.years_if_infected, rel=1e-12)
    assert burden.hiv_care > 10.0 * annual


def test_infection_shortens_life_in_discounted_years(burden):
    assert burden.years_if_infected < burden.years_if_uninfected


def test_a_higher_discount_rate_shrinks_the_case_for_prevention(params):
    """Prevention pays now and saves later, so discounting works against it.

    This is why the discount rate is a reported sensitivity and not a detail.
    """
    low = lifetime_burden_of_one_infection(params, 0.0)
    high = lifetime_burden_of_one_infection(params, 0.05)
    assert low.net_cost > high.net_cost
    assert low.qalys_lost > high.qalys_lost


def test_the_value_of_averting_rises_with_willingness_to_pay(burden):
    assert burden.value_at(150_000) > burden.value_at(50_000)
    assert burden.value_at(0.0) == pytest.approx(burden.net_cost, rel=1e-12)


def test_an_unsuppressed_infection_is_worse_health_but_not_a_bigger_bill(params):
    """Dying sooner is cheaper, which is the least comfortable true fact here.

    Anyone reading the cost column alone would conclude that failing to treat
    people is the better buy. It is not, and the QALY column is why.
    """
    sup = lifetime_burden_of_one_infection(params, 0.03, suppressed=True)
    unsup = lifetime_burden_of_one_infection(params, 0.03, suppressed=False)
    assert unsup.qalys_lost > sup.qalys_lost
    assert unsup.net_cost < sup.net_cost
