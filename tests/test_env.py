"""The environment is only worth having if four things are true of it.

It must be the same model (the default action has to reproduce the evaluation
exactly, or the agent is optimising a different thing from the one being
reported). Its reward has to be the quantity it claims to be. The action has to
be able to change the outcome, and only where the model says a policy could.
And the paired evaluation has to actually reduce variance, because without that
no comparison in this environment can be believed.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from prep_model import simulate
from prep_model.env import (ACTION_DIM, DEFAULT_ACTION, OBSERVATION_DIM, PrepEnv,
                            compare, constant_policy, default_policy, evaluate, seed_bank)


@pytest.fixture(scope="module")
def env(cfg, params):
    return PrepEnv(cfg, params, arm_id="C", wtp=100_000.0)


# -- it is the same model -----------------------------------------------------

def test_the_default_action_reproduces_the_evaluation_exactly(cfg, params, env):
    """Visits first, nobody prioritised: the policy the arms were costed under.

    If this ever drifts, every number the environment produces is about a model
    that is not the one in the report.
    """
    episode = env.run(default_policy, seed=cfg.seed)
    direct = simulate.run_arm(cfg, params, cfg.arm("C"))

    assert episode.result.epi["infections"] == direct.epi["infections"]
    assert episode.result.qalys == pytest.approx(direct.qalys, rel=1e-12)
    cats = cfg.perspectives[cfg.primary_perspective]
    assert episode.result.cost(cats) == pytest.approx(direct.cost(cats), rel=1e-12)
    assert episode.result.counters.doses == direct.counters.doses


def test_the_same_seed_and_actions_give_the_same_episode(env, cfg):
    a = env.run(default_policy, seed=cfg.seed)
    b = env.run(default_policy, seed=cfg.seed)
    assert a.ret == b.ret
    assert a.result.epi["infections"] == b.result.epi["infections"]


def test_a_different_seed_gives_a_different_episode(env, cfg):
    a = env.run(default_policy, seed=cfg.seed)
    b = env.run(default_policy, seed=cfg.seed + 1)
    assert a.ret != b.ret


# -- the reward is what it says it is ----------------------------------------

def test_the_return_is_the_net_benefit_the_evaluation_would_report(env, cfg):
    """Summed reward equals wtp * QALYs - cost, terminal value included."""
    episode = env.run(default_policy, seed=cfg.seed)
    cats = cfg.perspectives[cfg.primary_perspective]
    expected = (env.wtp * episode.result.qalys - episode.result.cost(cats)) * env.per_1000
    assert episode.ret == pytest.approx(expected, rel=1e-9), (
        "the reward stream and the ledger disagree, so the agent is being paid "
        "for something other than net monetary benefit"
    )


def test_the_terminal_value_is_paid_and_not_dropped(cfg, params):
    """Otherwise the last weeks of an episode are free and the agent knows it."""
    with_terminal = PrepEnv(cfg, params, arm_id="C").run(default_policy, seed=cfg.seed)
    without = PrepEnv(cfg.copy_with(time={"terminal_value": False}), params,
                      arm_id="C").run(default_policy, seed=cfg.seed)
    assert with_terminal.ret > without.ret


def test_a_higher_willingness_to_pay_raises_the_return(cfg, params):
    low = PrepEnv(cfg, params, arm_id="C", wtp=50_000.0).run(default_policy, seed=cfg.seed)
    high = PrepEnv(cfg, params, arm_id="C", wtp=150_000.0).run(default_policy, seed=cfg.seed)
    assert high.ret > low.ret


# -- the action can only act where the model says a policy could -------------

def _tight(cfg, params):
    """A configuration in which capacity binds but the programme still runs.

    The budgets are per 1000 and truncated to an integer, so on the 800-person
    test cohort a capacity of 0.5 per 1000 per week is a budget of zero: nobody
    ever starts, nothing is ever due, and every rationing test passes vacuously
    by comparing two empty programmes. 50 and 10 per 1000 give 40 contacts and
    8 initiations a week against roughly 225 people reachable, which is scarcity
    with something left to allocate.
    """
    return cfg, params.with_overrides({
        "capacity_contacts_per_1000_per_week": 50.0,
        "capacity_initiations_per_1000_per_week": 10.0,
        "reach_rate_annual_lowbarrier": 20.0,
    })


def test_priority_weights_change_the_outcome_when_capacity_binds(cfg, params):
    c, p = _tight(cfg, params)
    env = PrepEnv(c, p, arm_id="C")
    baseline = env.run(default_policy, seed=c.seed)
    # Serve the highest exposure group and the unsheltered first.
    targeted = env.run(constant_policy((1.0, 1.0, 1.0, 0.0, 0.0, 0.0)), seed=c.seed)
    assert targeted.result.counters.doses != baseline.result.counters.doses, (
        "the priority weights did nothing even though capacity was binding"
    )


def test_priority_weights_do_nothing_when_capacity_is_free(cfg, params):
    """Rationing is the only lever. With slots for everyone there is nothing to ration.

    A result that moved here would mean the action had found some other route
    into the model, which is exactly what must not happen.
    """
    p = params.with_overrides({"capacity_contacts_per_1000_per_week": 10_000.0,
                               "capacity_initiations_per_1000_per_week": 1_000.0})
    env = PrepEnv(cfg, p, arm_id="C")
    baseline = env.run(default_policy, seed=cfg.seed)
    targeted = env.run(constant_policy((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)), seed=cfg.seed)
    assert targeted.ret == pytest.approx(baseline.ret, rel=1e-12)


def test_the_budget_split_moves_visits_against_outreach(cfg, params):
    c, p = _tight(cfg, params)
    env = PrepEnv(c, p, arm_id="C")
    visits_first = env.run(default_policy, seed=c.seed)
    outreach_first = env.run(constant_policy((0.0,) + (0.0,) * 5), seed=c.seed)
    assert outreach_first.result.counters.contacts > visits_first.result.counters.contacts, (
        "reserving no capacity for scheduled visits did not free any for outreach"
    )


def test_an_action_of_the_wrong_length_is_an_error_not_a_silent_truncation(env, cfg):
    env.reset(cfg.seed)
    with pytest.raises(ValueError):
        env.step((1.0, 0.0))


# -- observations carry only what a programme could see ----------------------

def test_the_observation_is_finite_and_the_declared_shape(env, cfg):
    obs, _ = env.reset(cfg.seed)
    assert obs.shape == (OBSERVATION_DIM,)
    assert bool(torch.isfinite(obs).all())
    for _ in range(5):
        obs, reward, done, _, _ = env.step(DEFAULT_ACTION)
        assert bool(torch.isfinite(obs).all())
        assert math_isfinite(reward)
    assert len(DEFAULT_ACTION) == ACTION_DIM


def math_isfinite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


# -- the paired evaluation earns its keep ------------------------------------

def test_pairing_removes_most_of_the_noise(cfg, params):
    """The claim that makes any comparison here possible, checked rather than asserted.

    Two policies are compared on the same seeds. If the standard error of the
    paired difference were no smaller than the standard error of either policy
    alone, common random numbers would not be working and no difference this
    environment reported could be trusted.
    """
    c, p = _tight(cfg, params)
    env = PrepEnv(c, p, arm_id="C")
    seeds, _ = seed_bank(c.seed, 6)
    targeted = constant_policy((1.0, 1.0, 1.0, 0.0, 0.0, 0.0))

    a = evaluate(env, default_policy, seeds, "default")
    b = evaluate(env, targeted, seeds, "targeted")
    paired = compare(env, targeted, default_policy, seeds)

    unpaired_se = (a.se ** 2 + b.se ** 2) ** 0.5
    assert paired.se < 0.5 * unpaired_se, (
        f"paired se {paired.se:,.0f} against unpaired {unpaired_se:,.0f}: the seeds "
        "are not actually shared between the two policies"
    )
    assert paired.mean == pytest.approx(b.mean - a.mean, rel=1e-9)


def test_a_policy_identical_to_the_reference_shows_exactly_no_advantage(cfg, params):
    """The null case. Anything other than zero here is leakage between episodes."""
    env = PrepEnv(cfg, params, arm_id="C")
    seeds, _ = seed_bank(cfg.seed, 3)
    d = compare(env, constant_policy(DEFAULT_ACTION), default_policy, seeds)
    assert d.per_seed == [0.0, 0.0, 0.0]
    assert not d.is_an_improvement()


def test_the_seed_bank_holds_seeds_back(cfg):
    train, held = seed_bank(cfg.seed, 4, held_out=3)
    assert len(train) == 4 and len(held) == 3
    assert not (set(train) & set(held))


def test_an_evaluation_reports_a_standard_error_not_a_point(cfg, params):
    env = PrepEnv(cfg, params, arm_id="C")
    seeds, _ = seed_bank(cfg.seed, 4)
    e = evaluate(env, default_policy, seeds, "default")
    assert e.se > 0.0
    assert statistics.mean(e.returns) == pytest.approx(e.mean)
    assert "+/-" in str(e)
