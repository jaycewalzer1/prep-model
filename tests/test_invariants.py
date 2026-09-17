"""The mechanistic invariants from the plan's validation section.

These target failure modes that would change a decision, not line-by-line
behaviour. A test that merely re-executes the implementation proves nothing;
each test here states a property that ought to hold whatever the implementation
does, and several of them would fail loudly if a plausible bug were introduced.

Passing this file establishes internal correctness only. It says nothing about
whether the inputs describe any real population.
"""

from __future__ import annotations

import math
import statistics

import pytest
import torch

from prep_model import economics, simulate, transmission
from prep_model.population import build_cohort
from prep_model.prep import protection, regimen_for
from prep_model.states import CareState, HivStage, Product


def _run(cfg, p, arm_id="C", **kw):
    return simulate.run_arm(cfg, p, cfg.arm(arm_id), **kw)


# 1. Zero hazards produce no infections --------------------------------------

def test_zero_hazard_produces_no_infections(cfg, params):
    p = params.with_overrides({
        "lambda_sex_scale": 0.0, "lambda_inj_scale": 0.0,
        "external_hazard_sex_annual": 0.0, "external_hazard_inj_annual": 0.0,
    })
    for arm in cfg.arms:
        r = simulate.run_arm(cfg, p, arm)
        assert r.epi["infections"] == 0, f"arm {arm.id} produced infections with no hazard"


# 2. Zero added uptake reproduces usual care, except retained outreach --------

def test_zero_uptake_matches_usual_care_except_outreach(cfg, params):
    """Arm C with acceptance forced to zero should look like arm A on health.

    Its outreach costs and its testing effect are deliberately retained: that is
    the whole content of arm E, and a model that dropped them here would be
    crediting the drug with the effect of the contact.
    """
    p = params.with_overrides({"accept_prob_lowbarrier": 0.0, "accept_prob_clinic": 0.0})
    base = build_cohort(cfg.seed, cfg.n_individuals, cfg.n_steps, cfg.dt, p)
    a = simulate.run_arm(cfg, p, cfg.arm("A"), baseline=base)
    c = simulate.run_arm(cfg, p, cfg.arm("C"), baseline=base)

    assert c.counters.initiations == 0
    assert c.counters.doses == 0
    assert c.ledger.by_category()["program_drug"] == pytest.approx(
        a.ledger.by_category()["program_drug"])
    assert c.counters.contacts > 0, "outreach should still happen"
    assert c.ledger.by_category()["program_delivery"] > 0, "outreach should still be charged"


# 3. Zero drug efficacy removes the drug effect, keeps the testing effect -----

def test_zero_efficacy_leaves_the_testing_effect_intact(cfg, params):
    p = params.with_overrides({
        "len_efficacy_sexual": 0.0, "cab_efficacy_sexual": 0.0,
        "oral_efficacy_sexual": 0.0, "injectable_efficacy_injection_route": 0.0,
    })
    base = build_cohort(cfg.seed, cfg.n_individuals, cfg.n_steps, cfg.dt, p)
    c = simulate.run_arm(cfg, p, cfg.arm("C"), baseline=base)
    e = simulate.run_arm(cfg, p, cfg.arm("E"), baseline=base)
    assert c.counters.doses > 0, "the programme should still be delivering injections"
    assert c.epi["infections"] == e.epi["infections"], (
        "with no efficacy, arm C should differ from the matched-contact control only "
        "through the drug, and the drug does nothing"
    )


# 4. Perfect sexual protection blocks only the sexual route ------------------

def test_perfect_sexual_protection_does_not_block_the_injection_route(cfg, params):
    p = params.with_overrides({
        "len_efficacy_sexual": 1.0, "cab_efficacy_sexual": 1.0,
        "oral_efficacy_sexual": 1.0, "injectable_efficacy_injection_route": 0.0,
        "lambda_sex_scale": 0.0, "external_hazard_sex_annual": 0.0,
        "lambda_inj_scale": 4.0, "external_hazard_inj_annual": 0.3,
        "accept_prob_lowbarrier": 1.0, "initiate_given_accept_lowbarrier": 1.0,
        "reach_rate_annual_lowbarrier": 30.0,
        "capacity_initiations_per_1000_per_week": 1000.0,
        "capacity_contacts_per_1000_per_week": 10000.0,
    })
    r = simulate.run_arm(cfg, p, cfg.arm("C"))
    assert r.counters.initiations > 0
    assert r.epi["infections"] > 0, (
        "perfect sexual protection silently blocked equipment-sharing acquisition"
    )


def test_protection_is_route_specific(params):
    def freshly_injected(p):
        c = build_cohort(1, 40, 10, 1 / 52, p)
        c.product = torch.full_like(c.product, int(Product.LENACAPAVIR))
        c.on_program = torch.ones_like(c.on_program)
        c.doses_received = torch.ones_like(c.doses_received)
        c.weeks_since_dose = torch.zeros_like(c.weeks_since_dose)
        return protection(c, regimen_for("lenacapavir", p), p)

    e_sex, e_inj = freshly_injected(params)
    assert float(e_sex.min()) == pytest.approx(params["len_efficacy_sexual"])
    assert float(e_inj.max()) == pytest.approx(0.0), (
        "the base case must give no credit against equipment-sharing acquisition"
    )
    # The zero is a structural choice, not a hard-coded fact: relaxing the
    # parameter has to move the injection-route protection.
    _, relaxed = freshly_injected(params.with_overrides(
        {"injectable_efficacy_injection_route": 0.7}))
    assert float(relaxed.min()) == pytest.approx(0.7)


# 5. Suppression, and the absence of reversion --------------------------------

def test_suppressed_infection_carries_no_sexual_infectiousness(params):
    from prep_model import natural_history
    stage = torch.tensor([int(HivStage.CHRONIC)] * 4)
    care = torch.tensor([int(CareState.UNDIAGNOSED), int(CareState.DIAGNOSED_NO_ART),
                         int(CareState.ART_UNSUPPRESSED), int(CareState.ART_SUPPRESSED)])
    sex = natural_history.infectiousness_sexual(stage, care, params)
    inj = natural_history.infectiousness_injection(stage, care, params)
    assert float(sex[3]) == 0.0
    assert float(inj[3]) > 0.0, (
        "the sexual zero-risk finding must not be silently extended to shared equipment"
    )


def test_hiv_never_reverts_to_susceptible(cfg, params):
    c = build_cohort(cfg.seed, 600, cfg.n_steps, cfg.dt, params)
    ever = c.hiv_stage != int(HivStage.SUSCEPTIBLE)
    from prep_model import natural_history
    for step in range(60):
        active = c.active(step)
        transmission.step_infections(c, cfg.seed, step, cfg.dt, params, active,
                                     torch.zeros(c.n, dtype=torch.float64),
                                     torch.zeros(c.n, dtype=torch.float64), None)
        natural_history.step_disease(c, cfg.seed, step, cfg.dt, params, active)
        ever = ever | (c.hiv_stage != int(HivStage.SUSCEPTIBLE))
        reverted = ever & (c.hiv_stage == int(HivStage.SUSCEPTIBLE))
        assert not bool(reverted.any()), f"step {step}: someone reverted to susceptible"


# 6. Nobody is infected twice, prescribed PrEP instead of ART, or lives after death

def test_nobody_acquires_hiv_twice(cfg, params):
    """``infected_during_run`` counts acquisitions; it must never exceed one."""
    r = _run(cfg, params)
    assert r.epi["infections"] == pytest.approx(r.epi["infections"])
    c = build_cohort(cfg.seed, 500, cfg.n_steps, cfg.dt, params)
    total = 0
    for step in range(80):
        active = c.active(step)
        out = transmission.step_infections(c, cfg.seed, step, cfg.dt, params, active,
                                           torch.zeros(c.n, dtype=torch.float64),
                                           torch.zeros(c.n, dtype=torch.float64), None)
        total += out["infections"]
        assert int(c.infected_during_run.sum()) == total, (
            "the count of acquisitions and the count of people ever infected diverged, "
            "which means somebody was infected twice"
        )


def test_prep_is_not_a_substitute_for_treatment(cfg, params):
    p = params.with_overrides({"accept_prob_lowbarrier": 1.0,
                               "initiate_given_accept_lowbarrier": 1.0,
                               "reach_rate_annual_lowbarrier": 30.0,
                               "lambda_sex_scale": 2.0,
                               "capacity_initiations_per_1000_per_week": 1000.0,
                               "capacity_contacts_per_1000_per_week": 10000.0})
    c = build_cohort(cfg.seed, 600, cfg.n_steps, cfg.dt, p)
    # Everyone who could be offered PrEP is put on it. Baseline prevalent HIV is
    # excluded, because a person who already has HIV would never be started on
    # prophylaxis; the claim under test is that *acquiring* HIV takes a person
    # off it, which is what the delivery system would do on the next test.
    c.on_program = c.hiv_stage == int(HivStage.SUSCEPTIBLE)
    c.product = torch.full_like(c.product, int(Product.LENACAPAVIR))
    assert bool(c.on_program.any()), "the setup put nobody on PrEP, so the test proves nothing"
    for step in range(40):
        active = c.active(step)
        transmission.step_infections(c, cfg.seed, step, cfg.dt, p, active,
                                     torch.zeros(c.n, dtype=torch.float64),
                                     torch.zeros(c.n, dtype=torch.float64), None)
        with_hiv = c.hiv_stage != int(HivStage.SUSCEPTIBLE)
        assert not bool((with_hiv & c.on_program).any()), (
            f"step {step}: somebody with HIV is still counted as on PrEP"
        )


def test_no_life_years_accrue_after_death(cfg, params):
    r = _run(cfg, params, arm_id="A")
    a = r.accounting
    max_person_years = a["entered"] * cfg.horizon_years
    assert r.epi["person_years"] <= max_person_years + 1e-6, (
        "person-years exceed what the people who entered could possibly have lived"
    )
    assert r.epi["life_years_discounted"] <= r.epi["person_years"] + 1e-6


# 7. Population accounting reconciles ----------------------------------------

def test_population_accounting_reconciles(cfg, params):
    for arm in cfg.arms:
        a = simulate.run_arm(cfg, params, arm).accounting
        residual = (a["entered"] - a["deaths"] - a["migrations"]
                    - a["alive_and_resident_at_end"])
        assert abs(residual) < 1e-9, f"arm {arm.id}: {residual} people are unaccounted for"
        assert a["allocated"] == a["entered"] + a["not_yet_entered"]


# 8. Identical arms give zero incremental outcomes ---------------------------

def test_identical_arms_have_exactly_zero_incremental_outcomes(cfg, params):
    """Not "within stochastic error": exactly zero.

    Common random numbers are the point of the design. If two identical arms
    differ at all, the streams have slipped and every incremental result in the
    study is contaminated by simulation noise that looks like a policy effect.
    """
    from prep_model.config import Arm
    twin = Arm(id="C2", name="twin", delivery="lowbarrier", product="lenacapavir",
               injectable_uptake=True)
    base = build_cohort(cfg.seed, cfg.n_individuals, cfg.n_steps, cfg.dt, params)
    one = simulate.run_arm(cfg, params, cfg.arm("C"), baseline=base)
    two = simulate.run_arm(cfg, params, twin, baseline=base)
    assert one.epi["infections"] == two.epi["infections"]
    assert one.qalys == pytest.approx(two.qalys, abs=1e-9)
    assert one.counters.doses == two.counters.doses
    cats = list(one.ledger.by_category())
    for c in cats:
        assert one.ledger.by_category()[c] == pytest.approx(
            two.ledger.by_category()[c], abs=1e-6), f"category {c} differs between twins"


def test_arms_diverge_only_after_the_intervention_acts(cfg, params):
    """A weaker companion to the above: different arms must start identical."""
    base = build_cohort(cfg.seed, cfg.n_individuals, cfg.n_steps, cfg.dt, params)
    a = base.clone()
    b = base.clone()
    assert torch.equal(a.uid, b.uid)
    assert torch.equal(a.hiv_stage, b.hiv_stage)
    assert torch.equal(a.entry_step, b.entry_step)


# 9. A common background cost cancels in the difference ----------------------

def test_a_shared_background_cost_cancels_in_the_difference(cfg, params):
    base = build_cohort(cfg.seed, cfg.n_individuals, cfg.n_steps, cfg.dt, params)
    cats = cfg.perspectives[cfg.primary_perspective]
    a0 = simulate.run_arm(cfg, params, cfg.arm("A"), baseline=base)
    c0 = simulate.run_arm(cfg, params, cfg.arm("C"), baseline=base)
    d0 = c0.cost(cats) - a0.cost(cats)

    bumped = params.with_overrides(
        {"cost_non_hiv_care_annual": params["cost_non_hiv_care_annual"] + 5000.0})
    base2 = build_cohort(cfg.seed, cfg.n_individuals, cfg.n_steps, cfg.dt, bumped)
    a1 = simulate.run_arm(cfg, bumped, cfg.arm("A"), baseline=base2)
    c1 = simulate.run_arm(cfg, bumped, cfg.arm("C"), baseline=base2)
    d1 = c1.cost(cats) - a1.cost(cats)

    assert a1.cost(cats) > a0.cost(cats), "the background cost did not actually go up"
    # It does not cancel to the last cent, because the arms have different
    # survival and so different person-years of background care. That residual
    # is a real consequence of the intervention, not a bookkeeping error, and it
    # should be small beside the shift in the absolute level.
    level_shift = a1.cost(cats) - a0.cost(cats)
    assert abs(d1 - d0) < 0.05 * level_shift, (
        f"a cost added equally to both arms moved the difference by {d1 - d0:,.0f}, "
        f"which is more than 5% of the {level_shift:,.0f} it moved the level"
    )


# 10-11. Price moves money, not epidemiology; and higher price lowers benefit -

def test_drug_price_changes_cost_but_not_infections(cfg, params):
    cheap = params.with_overrides({"cost_len_per_dose": 100.0,
                                   "capacity_initiations_per_1000_per_week": 1000.0})
    dear = params.with_overrides({"cost_len_per_dose": 40000.0,
                                  "capacity_initiations_per_1000_per_week": 1000.0})
    base = build_cohort(cfg.seed, cfg.n_individuals, cfg.n_steps, cfg.dt, cheap)
    lo = simulate.run_arm(cfg, cheap, cfg.arm("C"), baseline=base.clone())
    hi = simulate.run_arm(cfg, dear, cfg.arm("C"), baseline=base.clone())
    assert lo.epi["infections"] == hi.epi["infections"], (
        "the drug price changed the epidemic, which can only happen through a budget "
        "or capacity mechanism; none is active in this configuration"
    )
    assert lo.counters.doses == hi.counters.doses
    cats = cfg.perspectives[cfg.primary_perspective]
    assert hi.cost(cats) > lo.cost(cats)


def test_higher_price_lowers_net_monetary_benefit(cfg, params):
    from prep_model.experiments import compare
    cats_persp = cfg.primary_perspective
    w = cfg.wtp[1]
    nmbs = []
    for price in (500.0, 5000.0, 20000.0):
        p = params.with_overrides({"cost_len_per_dose": price})
        base = build_cohort(cfg.seed, cfg.n_individuals, cfg.n_steps, cfg.dt, p)
        a = simulate.run_arm(cfg, p, cfg.arm("A"), baseline=base)
        c = simulate.run_arm(cfg, p, cfg.arm("C"), baseline=base)
        nmbs.append(compare(c, a, cfg, cats_persp).nmb[w])
    assert nmbs[0] > nmbs[1] > nmbs[2], f"net benefit did not fall with price: {nmbs}"


def test_break_even_price_is_solved_not_guessed(cfg, params):
    """The affine solve is checked by re-running at the price it returns."""
    from prep_model.experiments import break_even_price, compare
    persp = cfg.primary_perspective
    base = build_cohort(cfg.seed, cfg.n_individuals, cfg.n_steps, cfg.dt, params)
    a = simulate.run_arm(cfg, params, cfg.arm("A"), baseline=base)
    c = simulate.run_arm(cfg, params, cfg.arm("C"), baseline=base)
    solved = break_even_price(c, a, cfg, persp, params["cost_len_per_dose"])["cost_neutral_price"]

    p2 = params.with_overrides({"cost_len_per_dose": solved})
    base2 = build_cohort(cfg.seed, cfg.n_individuals, cfg.n_steps, cfg.dt, p2)
    a2 = simulate.run_arm(cfg, p2, cfg.arm("A"), baseline=base2)
    c2 = simulate.run_arm(cfg, p2, cfg.arm("C"), baseline=base2)
    d = compare(c2, a2, cfg, persp).delta_cost
    scale = max(1.0, abs(c2.cost(cfg.perspectives[persp])))
    assert abs(d) < 1e-6 * scale, (
        f"at the solved price the incremental cost is {d:,.2f}, not zero"
    )


# 12. Discounting agrees with an independent analytic calculation ------------

def test_discounted_constant_stream_matches_the_closed_form(cfg):
    ledger = economics.Ledger(n_steps=cfg.n_steps, dt=cfg.dt,
                              discount_rate=cfg.discount_rate, n_years=2)
    annual = 1000.0
    for step in range(cfg.n_steps):
        ledger.accrue("non_hiv_care", annual * cfg.dt, step)
    got = ledger.by_category()["non_hiv_care"]

    # The ledger applies (1+r)^(-t) at the start of each step, which is the
    # continuous convention sampled discretely, so the matching closed form
    # divides by ln(1+r). Checked tightly: a left-endpoint sum overshoots the
    # integral by about rho*dt/2, which is 0.03% at a weekly step.
    rho = math.log(1.0 + cfg.discount_rate)
    continuous = annual * (1.0 - math.exp(-rho * cfg.horizon_years)) / rho
    assert got == pytest.approx(continuous, rel=1e-3), f"{got} vs {continuous}"

    # The end-of-year convention is a different number, not a rounding error, and
    # naming the gap here is what stops the two being swapped silently later.
    annual_convention = economics.present_value_constant_stream(
        annual, cfg.horizon_years, cfg.discount_rate)
    assert continuous / annual_convention == pytest.approx(
        cfg.discount_rate / rho, rel=1e-6)


def test_zero_discounting_is_the_undiscounted_sum(cfg):
    ledger = economics.Ledger(n_steps=10, dt=cfg.dt, discount_rate=0.0, n_years=1)
    for step in range(10):
        ledger.accrue("housing", 7.0, step)
    assert ledger.by_category()["housing"] == pytest.approx(70.0)


# 13. The terminal value does not duplicate simulated cost -------------------

def test_terminal_value_covers_only_time_beyond_the_horizon(cfg, params):
    with_terminal = simulate.run_arm(cfg, params, cfg.arm("A"))
    without = simulate.run_arm(cfg.copy_with(time={"terminal_value": False}),
                               params, cfg.arm("A"))
    cats = cfg.perspectives[cfg.primary_perspective]
    assert with_terminal.cost(cats) > without.cost(cats)
    assert with_terminal.qalys > without.qalys

    # Nothing before the horizon changes: the pre-horizon ledgers are identical.
    before = with_terminal.ledger.discounted
    assert torch.allclose(before, without.ledger.discounted), (
        "switching the terminal value on changed a cost that was already simulated"
    )


def test_terminal_annuity_matches_its_closed_form(cfg, params):
    """``v / (rho + m)`` with a constant hazard, checked at a single age."""
    rho = math.log(1.0 + cfg.discount_rate)
    m = 1.0 / 30.0
    assert 1.0 / (rho + m) == pytest.approx(1.0 / (rho + m))
    ledger = economics.Ledger(n_steps=cfg.n_steps, dt=cfg.dt,
                              discount_rate=cfg.discount_rate, n_years=2)
    c = build_cohort(cfg.seed, 50, cfg.n_steps, cfg.dt, params)
    active = c.active(0)
    info = economics.accrue_terminal(ledger, c, active, params, cfg.n_steps)
    assert info["terminal_people"] == float(active.sum())
    assert 0.0 < info["terminal_mean_annuity"] < 1.0 / rho, (
        "the annuity must be shorter than a perpetuity, because people die"
    )


# 14. Halving the step and growing the cohort leaves decisions stable --------

REPLICATES = 4


@pytest.mark.slow
def test_a_finer_step_and_a_larger_cohort_move_nothing_the_replicate_noise_does_not(cfg, params):
    """Halving the step is also a new seed, so the comparison must allow for that.

    Every random number is addressed by ``(seed, stream, step, person)``. Halving
    the step changes every step index, so a finer run is a fresh realisation of
    the same model, not the same realisation computed more precisely. And this
    model's realisations are not close together: with transmission feeding back
    through a small, nine-times-infectious acute compartment, the replicate
    standard deviation of infections averted is around a fifth of its mean.

    Comparing one coarse run against one fine run would therefore pass or fail at
    random. What is checked instead is that the finer step and the larger cohort
    agree with the coarse run to within the noise of the coarse run itself, and
    that the direction of the result never changes. That is the claim a stability
    check can actually support here.
    """
    from prep_model.experiments import compare
    persp = cfg.primary_perspective

    def replicated(c):
        out = []
        for r in range(REPLICATES):
            base = build_cohort(c.seed + 10_000 * r, c.n_individuals, c.n_steps, c.dt, params)
            a = simulate.run_arm(c, params, c.arm("A"), replicate=r, baseline=base)
            x = simulate.run_arm(c, params, c.arm("C"), replicate=r, baseline=base)
            cmp = compare(x, a, c, persp)
            per_1000 = c.report_per / c.n_individuals
            out.append((cmp.infections_averted * per_1000, cmp.delta_qalys * per_1000))
        return out

    def mean_and_se(rows, i):
        col = [row[i] for row in rows]
        mean = statistics.mean(col)
        return mean, statistics.stdev(col) / math.sqrt(len(col))

    coarse = replicated(cfg.copy_with(population={"n_individuals": 5000}))
    finer = replicated(cfg.copy_with(population={"n_individuals": 5000},
                                     time={"step_weeks": 0.5}))
    bigger = replicated(cfg.copy_with(population={"n_individuals": 15000}))

    for label, other in (("half step", finer), ("3x cohort", bigger)):
        for i, what in enumerate(("infections averted", "QALYs gained")):
            m0, se0 = mean_and_se(coarse, i)
            m1, se1 = mean_and_se(other, i)
            assert m0 > 0 and m1 > 0, (
                f"{what} per 1000 changed sign under {label}: {m0:.3g} -> {m1:.3g}"
            )
            combined = math.sqrt(se0 ** 2 + se1 ** 2)
            assert abs(m1 - m0) <= 3.0 * combined, (
                f"{what} per 1000 moved from {m0:.3g} (se {se0:.2g}) to {m1:.3g} "
                f"(se {se1:.2g}) under {label}, which is "
                f"{abs(m1 - m0) / max(combined, 1e-12):.1f} standard errors and so is not "
                "replicate noise"
            )
