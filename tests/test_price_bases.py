"""The break-even price is a different number under each accounting basis.

The point of reporting three is that a reader can see how much of the answer is
a fact about the drug and how much is a choice about whose budget is being
protected. That only works if the bases are genuinely ordered and genuinely
different, and if the lifetime figures replace the simulation's credit for an
averted infection rather than being added on top of it.
"""

from __future__ import annotations

import pytest

from prep_model.analytic import break_even_price_bases, lifetime_burden_of_one_infection


@pytest.fixture(scope="module")
def burden(params):
    return lifetime_burden_of_one_infection(params, 0.03)


def _basis(burden, **kw):
    base = dict(arm_id="C", label="outreach", current_price=14_109.0,
                discounted_doses=21_441.0, infections_averted=253.0,
                programme_cost=120_000_000.0, budget_price_as_modelled=-443.0,
                burden=burden, wtp=100_000.0)
    base.update(kw)
    return break_even_price_bases(**base)


def test_crediting_a_whole_lifetime_raises_the_price_a_payer_can_justify(burden):
    """The simulation stops counting when a person leaves; the lifetime figure
    does not. It must therefore support a higher price, never a lower one."""
    b = _basis(burden)
    assert b.budget_price_lifetime > b.budget_price_as_modelled


def test_health_is_worth_something_so_the_threshold_beats_the_budget(burden):
    b = _basis(burden)
    assert b.threshold_price_lifetime > b.budget_price_lifetime


def test_the_three_bases_are_ordered(burden):
    b = _basis(burden)
    assert (b.budget_price_as_modelled
            < b.budget_price_lifetime
            < b.threshold_price_lifetime)


def test_a_worthless_qaly_collapses_the_threshold_onto_the_budget(burden):
    """At a willingness to pay of zero the only thing left is the money."""
    b = _basis(burden, wtp=0.0)
    assert b.threshold_price_lifetime == pytest.approx(b.budget_price_lifetime, rel=1e-12)


def test_averting_nothing_means_the_programme_is_pure_cost(burden):
    """With no infections averted the break-even price is what you get by
    subtracting the whole programme bill from the price of the doses.

    It need not be negative -- that depends on how the delivery bill compares
    with the dose count -- but it must fall below what the dose costs today,
    because nothing has been bought with the difference.
    """
    b = _basis(burden, infections_averted=0.0)
    expected = 14_109.0 - 120_000_000.0 / 21_441.0
    assert b.budget_price_lifetime == pytest.approx(expected, rel=1e-12)
    assert b.budget_price_lifetime < b.current_price


def test_the_discount_needed_is_negative_when_the_arm_already_pays(burden):
    """A negative discount is the readable way to say 'you could pay more'."""
    cheap = _basis(burden, programme_cost=1_000_000.0)
    assert cheap.threshold_price_lifetime > cheap.current_price
    assert cheap.discount_required < 0.0


def test_more_doses_for_the_same_benefit_lowers_the_price_each_can_command(burden):
    lean = _basis(burden, discounted_doses=21_441.0)
    fat = _basis(burden, discounted_doses=60_000.0)
    assert fat.threshold_price_lifetime < lean.threshold_price_lifetime


def test_an_arm_that_buys_no_doses_has_no_break_even_price(burden):
    """Dividing by a zero dose count would invent a number. It must refuse."""
    with pytest.raises(ValueError, match="no doses"):
        _basis(burden, discounted_doses=0.0)
