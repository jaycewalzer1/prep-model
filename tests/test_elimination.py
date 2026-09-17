"""Can the programme drive incidence to zero, and does the ladder say so honestly?

The decomposition at the top rung is the part worth protecting. It is easy to
write a coverage sweep that appears to approach elimination simply because the
residual is never attributed to anything, and a reader would take that as a
statement about what a programme could achieve. These checks require the floor to
be real: saturating delivery must not remove infections that arrive from outside
the population, and must not remove infections by a route the product is assumed
not to protect against.
"""

from __future__ import annotations

import pytest

from prep_model.elimination import SATURATED, ladder


@pytest.fixture(scope="module")
def small(cfg, params):
    """Two rungs only. The shape of the result is what is under test, not its level."""
    return ladder(cfg, params, arm_id="C", multiples=(1.0, 8.0), reps=1)


def test_more_outreach_does_not_increase_incidence(small):
    assert small.rungs[-1].infections <= small.rungs[0].infections


def test_saturating_delivery_leaves_infections_behind(small):
    """If this ever hit zero, the model would be claiming elimination is a
    delivery problem. It is not: there is an external hazard and an unprotected
    route, and both survive an unlimited programme."""
    assert small.rungs[-1].infections > 0


def test_the_external_hazard_is_a_floor_the_programme_cannot_reach(small):
    """Removing the hazard that arrives from outside must remove infections that
    saturating the programme could not."""
    saturated = next(f for f in small.floors if f.label == "saturated programme")
    closed = next(f for f in small.floors if f.label == "+ no external hazard")
    assert closed.infections < saturated.infections


def test_the_floors_only_ever_go_down(small):
    """Each row removes a barrier and keeps the ones above it, so the sequence is
    cumulative and must be monotone. A rise would mean the overrides interact."""
    counts = [f.infections for f in small.floors]
    assert counts == sorted(counts, reverse=True)


def test_the_saturated_capacity_really_is_unbinding(cfg, params):
    p = params.with_overrides(SATURATED)
    assert p["capacity_contacts_per_1000_per_week"] >= 10_000.0
    assert p["capacity_initiations_per_1000_per_week"] >= 1_000.0


def test_coverage_never_reaches_everyone(small):
    """Refusal, failure to initiate and lapse are not capacity problems, so an
    unlimited programme still does not put everyone on the product."""
    assert 0.0 < small.rungs[-1].on_program_share < 1.0


def test_the_reference_arm_is_worse_than_any_rung(small):
    assert small.reference_infections > small.rungs[-1].infections
