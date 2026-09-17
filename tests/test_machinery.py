"""The machinery the invariants depend on: draws, the registry, and the ledger.

If the random streams are not actually stateless, or the price index is not
actually applied, every invariant above passes for the wrong reason.
"""

from __future__ import annotations

import math

import pytest
import torch

from prep_model.experiments import DOMINANT, DOMINATED, efficiency_frontier, icer
from prep_model.params import ParameterError
from prep_model.rng import Stream, categorical, normal, uniform


# -- the random draws ---------------------------------------------------------

def test_draws_depend_only_on_the_coordinate():
    uid = torch.arange(1000)
    a = uniform(7, Stream.INFECTION, 12, uid)
    b = uniform(7, Stream.INFECTION, 12, uid)
    assert torch.equal(a, b), "the same coordinate gave two different numbers"


def test_draws_are_uniform_and_streams_are_independent():
    uid = torch.arange(200_000)
    u = uniform(3, Stream.EXIT, 5, uid)
    assert 0.0 <= float(u.min()) and float(u.max()) < 1.0
    assert float(u.mean()) == pytest.approx(0.5, abs=0.005)
    assert float(u.var()) == pytest.approx(1 / 12, abs=0.002)

    v = uniform(3, Stream.TESTING, 5, uid)
    corr = float(((u - u.mean()) * (v - v.mean())).mean() / (u.std() * v.std()))
    assert abs(corr) < 0.01, f"two event streams are correlated at {corr:.4f}"


def test_changing_any_coordinate_changes_the_draw():
    uid = torch.arange(500)
    base = uniform(1, Stream.EXIT, 2, uid)
    for other in (uniform(2, Stream.EXIT, 2, uid),
                  uniform(1, Stream.TESTING, 2, uid),
                  uniform(1, Stream.EXIT, 3, uid)):
        assert not torch.equal(base, other)


def test_normal_draws_are_standard():
    uid = torch.arange(200_000)
    z = normal(11, Stream.ATTR_AGE, 0, uid)
    assert float(z.mean()) == pytest.approx(0.0, abs=0.01)
    assert float(z.std()) == pytest.approx(1.0, abs=0.01)


def test_categorical_respects_the_probabilities():
    n = 100_000
    u = uniform(5, Stream.HOUSING_DEST, 1, torch.arange(n))
    probs = torch.tensor([[0.2, 0.5, 0.3]] * n, dtype=torch.float64)
    out = categorical(u, probs)
    for k, want in enumerate((0.2, 0.5, 0.3)):
        assert float((out == k).to(torch.float64).mean()) == pytest.approx(want, abs=0.01)


# -- the registry -------------------------------------------------------------

def test_the_registry_has_no_structural_problems(registry):
    problems = registry.structural_audit()
    assert problems == [], "\n".join(problems)


def test_no_row_claims_to_be_a_local_measurement_without_a_source(registry):
    """The flag is the claim, so the flag is what has to be earned.

    Most rows here carry no URL: they are transported from general knowledge of
    the literature or are outright placeholders, and the report says so. The one
    flag that cannot be asserted without a source is ``observed_local``, because
    that is the flag that would let a reader believe the number was measured in
    the study city.
    """
    flags = {"observed_local", "transported", "calibrated", "hypothetical"}
    for name, row in registry.rows.items():
        assert row.assumption_flag in flags, f"{name} carries an unknown flag"
        assert row.evidence_grade in set("ABCD"), f"{name} carries an unknown grade"
        if row.assumption_flag == "observed_local":
            assert row.source_url, f"{name} claims a local observation with nothing behind it"


def test_the_unsourced_share_is_declared_rather_than_discovered(registry):
    """A count the report prints, pinned so it cannot drift silently upward."""
    unsourced = [n for n, r in registry.rows.items() if not r.source_url]
    assert len(unsourced) / len(registry.rows) > 0.5, (
        "most rows used to be unsourced; if that is no longer true, update the "
        "report's evidence section rather than this test"
    )
    # The report names the hypothetical inputs the run actually read, so read them
    # all and check the two lists are the same set.
    p = registry.resolve(2025)
    for name in registry.rows:
        p[name]
    hypothetical = {n for n, r in registry.rows.items() if r.assumption_flag == "hypothetical"}
    assert set(p.hypothetical_inputs()) == hypothetical, (
        "the flag and the list the report prints have come apart, so a placeholder "
        "could be used without being named as one"
    )


def test_quantiles_are_monotone_and_inside_the_stated_bounds(registry):
    for name, row in registry.rows.items():
        if row.distribution == "fixed":
            assert row.quantile(0.01) == row.quantile(0.99) == row.value
            continue
        qs = [row.quantile(u) for u in (0.01, 0.25, 0.5, 0.75, 0.99)]
        assert qs == sorted(qs), f"{name}: quantile function is not monotone"
        if row.distribution in {"uniform", "triangular"}:
            assert row.lower <= qs[0] and qs[-1] <= row.upper, f"{name}: outside its bounds"


def test_money_is_converted_to_the_declared_price_year(registry):
    p2015 = registry.resolve(2015)
    p2025 = registry.resolve(2025)
    money = [n for n, r in registry.rows.items() if r.is_money and r.currency_year
             and int(float(r.currency_year)) != 2015 and r.value != 0]
    assert money, "no monetary rows to check"
    for name in money:
        assert p2015[name] != p2025[name], f"{name} was not deflated between price years"


def test_an_unknown_parameter_is_an_error_not_a_zero(registry):
    p = registry.resolve(2025)
    with pytest.raises(ParameterError):
        p["a_parameter_nobody_defined"]


def test_an_override_must_name_a_real_parameter(registry):
    p = registry.resolve(2025)
    with pytest.raises(ParameterError):
        p.with_overrides({"not_a_parameter": 1.0})


def test_the_registry_records_what_the_model_read(registry):
    p = registry.resolve(2025)
    assert p.accessed == set()
    p["cost_hiv_test"]
    assert "cost_hiv_test" in p.accessed
    assert "cost_hiv_test" not in p.unused()


# -- incremental arithmetic ---------------------------------------------------

def test_icer_names_dominance_rather_than_printing_an_ambiguous_ratio():
    assert icer(-100.0, 2.0) == DOMINANT
    assert icer(100.0, -2.0) == DOMINATED
    assert icer(100.0, 0.0) == DOMINATED
    assert icer(-100.0, 0.0) == DOMINANT
    assert icer(100.0, 2.0) == pytest.approx(50.0)


def test_the_frontier_removes_simple_and_extended_dominance():
    class Fake:
        def __init__(self, arm_id, cost, qalys):
            self.arm_id = arm_id
            self.arm_name = arm_id
            self._c, self._q = cost, qalys

        def cost(self, _cats):
            return self._c

        @property
        def qalys(self):
            return self._q

    class FakeCfg:
        perspectives = {"p": ["hiv_care"]}

    arms = {
        "A": Fake("A", 0.0, 0.0),
        "B": Fake("B", 100.0, 1.0),     # ICER 100 vs A
        "C": Fake("C", 150.0, 1.2),     # ICER 250 vs B, then 150 vs A to D: extended
        "D": Fake("D", 200.0, 3.0),     # ICER 50 vs A: makes C extended-dominated
        "E": Fake("E", 250.0, 0.5),     # costlier and worse than B: simply dominated
    }
    out = efficiency_frontier(arms, FakeCfg(), "p")
    status = {d["arm_id"]: d["status"] for d in out}
    assert status["E"] == "dominated"
    assert status["C"] == "extended_dominated"
    assert status["A"] == "on_frontier"
    assert status["D"] == "on_frontier"


def test_net_monetary_benefit_and_the_icer_agree_at_the_threshold():
    d_cost, d_qaly = 100_000.0, 2.0
    ratio = icer(d_cost, d_qaly)
    assert ratio * d_qaly - d_cost == pytest.approx(0.0), (
        "net benefit at the ICER itself must be zero, by definition"
    )


# -- the analytic prototype ---------------------------------------------------

def test_the_break_even_approximation_states_its_own_error():
    from prep_model.analytic import break_even, exact_infections_averted
    for risk in (0.001, 0.02, 0.10):
        approx = break_even(risk, 0.9, 400_000.0, participants=1000).infections_averted
        exact = exact_infections_averted(risk, 0.9, 1000)
        rel = abs(approx - exact) / exact
        assert rel < 0.10, f"at risk {risk} the small-risk approximation is off by {rel:.1%}"
        if risk <= 0.001:
            assert rel < 0.001


def test_housing_savings_are_zero_unless_someone_supplies_them():
    from prep_model.analytic import break_even
    a = break_even(0.02, 0.9, 400_000.0, participants=1000)
    b = break_even(0.02, 0.9, 400_000.0, participants=1000,
                   hiv_attributable_housing_saving=50_000.0)
    assert b.lifetime_cost_offset > a.lifetime_cost_offset
    assert a.lifetime_cost_offset == pytest.approx(
        a.infections_averted * 400_000.0), (
        "the default must not quietly credit a housing saving nobody sourced"
    )


# -- the deterministic stage --------------------------------------------------

def test_the_deterministic_model_conserves_people(params):
    from prep_model import deterministic
    out = deterministic.run(params, 3.0, 1 / 52, n_people=1000.0)
    assert not math.isnan(out.prevalence)
    assert 0.0 <= out.prevalence <= 1.0
    assert 0.0 <= out.diagnosed_prevalence <= out.prevalence + 1e-9
    assert 0.0 <= out.suppressed_among_diagnosed <= 1.0
    assert out.share_unsheltered == pytest.approx(
        deterministic.run(params, 3.0, 1 / 52, n_people=5000.0).share_unsheltered, rel=1e-6), (
        "an output stated as a share moved when the population was rescaled"
    )


def test_the_deterministic_batch_dimension_agrees_with_one_at_a_time(params):
    from prep_model import deterministic
    name = "lambda_sex_scale"
    values = [0.03, 0.09]
    batched = deterministic.run(params, 2.0, 1 / 52, batch={
        name: torch.tensor(values, dtype=torch.float64)}).as_dict()
    for i, v in enumerate(values):
        single = deterministic.run(params.with_overrides({name: v}), 2.0, 1 / 52).as_dict()
        for key in single:
            assert float(batched[key][i]) == pytest.approx(float(single[key][0]), rel=1e-9), (
                f"{key} differs between the batched and the single run at {name}={v}"
            )
