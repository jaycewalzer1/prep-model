"""A reinforcement learning environment over the same simulation, for one
question the cost-effectiveness analysis cannot answer.

The economic evaluation compares five *fixed* arms. It cannot say who should be
served first when the week's capacity binds, because that is a within-arm
sequencing decision and the arms hold it fixed. Two places in ``delivery.py``
make that decision by fiat, and both are flagged in its own docstring as policy
choices rather than neutral defaults: scheduled injection visits are served
before new outreach, and within each queue the people served are chosen at
random. This environment turns exactly those two decisions into an action and
leaves the rest of the model alone.

The environment is deliberately not the deliverable. Three things are true of it
and are stated here rather than discovered later.

**The noise floor is high.** The replicate standard deviation of infections
averted is around a fifth of its mean, because transmission feeds back through a
small and highly infectious acute compartment. A learning curve from a single
seed is a picture of that noise. ``evaluate`` and ``compare`` below therefore
run a fixed bank of seeds and report a standard error, and ``compare`` pairs the
seeds so that the difference between two policies is measured on identical
random draws. A policy that is not better than the default by more than a couple
of standard errors on *held-out* seeds has not been shown to be better at all.

**The model is under-determined.** Four of ten calibrated parameters are flagged
as not identified by the available targets. An optimiser is an efficient way of
finding the region where a model is least constrained by evidence. That is the
reason the action space here is small and interpretable: a weight per observable
feature, which can be read and argued with, rather than a network whose
behaviour is only visible through its outputs.

**The agent sees what a delivery programme would see.** The observation carries
coverage, lapses, capacity pressure and testing yield. It does not carry true
prevalence, true incidence, or who is infected. A policy trained on those would
not be implementable and its advantage would be an artefact.

Reward is net monetary benefit accrued in the week, per 1,000 eligible adults,
at a stated willingness to pay: ``wtp * QALYs - cost``, over a stated
perspective, discounted on the same schedule as the ledger. It is an absolute
quantity, not an increment against usual care. Under common random numbers the
*difference* in episode return between two policies on the same seed is exactly
the incremental net benefit the evaluation reports, which is why no paired
control arm has to be simulated alongside.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .config import Config
from .delivery import DeliveryPlan
from .params import Params
from .simulate import ArmResult, Simulation
from .states import Housing, N_EXPOSURE_GROUPS

PRIORITY_FEATURES = (
    "exposure_group",      # risk category from a standard assessment, scaled to [0, 1]
    "unsheltered",         # 1 if sleeping outside
    "shares_equipment",    # 1 if injecting and sharing, as disclosed
    "overdue",             # how far past the due date, capped at one interval
    "lapsed",              # 1 if a due visit was already missed
)
"""What a priority score may be built from.

Every one of these is something an outreach worker could know at the door. The
latent engagement propensity that drives retention in the model is deliberately
absent: it is not observable, and a policy that used it would be a policy that
cannot be run.
"""

ACTION_DIM = 1 + len(PRIORITY_FEATURES)
OBSERVATION_DIM = 9 + N_EXPOSURE_GROUPS

DEFAULT_ACTION = (1.0,) + (0.0,) * len(PRIORITY_FEATURES)
"""Visits before outreach, nobody prioritised: the model's own hard-coded policy.

An episode run with this action at every step reproduces ``simulate.run_arm``
exactly, which a test asserts.
"""


def _feature_matrix(c, regimen) -> torch.Tensor:
    """The priority features, one column per person, each scaled to [0, 1]."""
    cols = [
        c.exposure_group.to(torch.float64) / max(1, N_EXPOSURE_GROUPS - 1),
        (c.housing == int(Housing.UNSHELTERED)).to(torch.float64),
        c.shares_equipment.to(torch.float64),
        torch.zeros_like(c.age) if regimen is None else
        (c.weeks_since_dose / regimen.interval_weeks).clamp(0.0, 1.0),
        c.lapsed.to(torch.float64),
    ]
    return torch.stack(cols)


@dataclass
class EpisodeSummary:
    """What one episode produced, in the units the evaluation report uses."""

    seed: int
    ret: float
    result: ArmResult
    per_1000: float

    @property
    def infections(self) -> float:
        return self.result.epi["infections"] * self.per_1000

    @property
    def qalys(self) -> float:
        return self.result.qalys * self.per_1000


class PrepEnv:
    """One episode is one arm run to the horizon, a week at a time.

    Not a ``gymnasium.Env`` subclass, to avoid the dependency, but the same
    shape: ``reset() -> (obs, info)`` and ``step(action) -> (obs, reward,
    terminated, truncated, info)``. Observations and actions are 1-D float64
    tensors.
    """

    action_dim = ACTION_DIM
    observation_dim = OBSERVATION_DIM

    def __init__(self, cfg: Config, p: Params, arm_id: str = "C",
                 wtp: float = 100_000.0, perspective: str | None = None):
        self.cfg = cfg
        self.p = p
        self.arm = cfg.arm(arm_id)
        self.wtp = float(wtp)
        self.perspective = perspective or cfg.primary_perspective
        self.categories = cfg.perspectives[self.perspective]
        self.per_1000 = cfg.report_per / cfg.n_individuals
        self.sim: Simulation | None = None
        self._last: dict[str, float] = {}

    # -- the loop -------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg if seed is None else self.cfg.copy_with(seed=seed)
        self.sim = Simulation.start(cfg, self.p, self.arm)
        self._last = {"blocked": 0.0, "initiations": 0.0, "doses": 0.0,
                      "diagnoses": 0.0, "contacts": 0.0}
        return self._observe(), {"seed": self.sim.seed}

    def step(self, action: Sequence[float]) -> tuple[torch.Tensor, float, bool, bool, dict]:
        if self.sim is None:
            raise RuntimeError("reset() before step()")
        if len(action) != ACTION_DIM:
            raise ValueError(f"action has {len(action)} entries, expected {ACTION_DIM}: "
                             f"one budget split and one weight per {PRIORITY_FEATURES}")

        self.sim.plan = self._plan(action)
        cost_before = self.sim.ledger.total_discounted(self.categories)
        qalys_before = self.sim.ledger.total_qalys()
        info = self.sim.advance()
        terminated = self.sim.done

        result = None
        if terminated:
            # The terminal value is part of the last reward. Without it a policy
            # is paid to let people reach the horizon in any state at all.
            result = self.sim.finish()
        reward = self._reward(cost_before, qalys_before)

        self._last = {"blocked": info["delivery_capacity_blocked_contacts"]
                      + info["delivery_capacity_blocked_initiations"],
                      "initiations": info["delivery_initiations"],
                      "doses": info["delivery_doses"],
                      "diagnoses": info["delivery_program_diagnoses"],
                      "contacts": info["delivery_contacts"]}
        if result is not None:
            info["result"] = result
        return self._observe(), reward, terminated, False, info

    def run(self, policy: Callable[[torch.Tensor], Sequence[float]],
            seed: int | None = None) -> EpisodeSummary:
        """One whole episode under ``policy``, returning the return and the arm."""
        obs, _ = self.reset(seed)
        total = 0.0
        result: ArmResult | None = None
        while True:
            obs, reward, terminated, _, info = self.step(policy(obs))
            total += reward
            if terminated:
                result = info["result"]
                break
        assert result is not None
        return EpisodeSummary(seed=self.sim.seed, ret=total, result=result,
                              per_1000=self.per_1000)

    # -- the pieces -----------------------------------------------------------

    def _plan(self, action: Sequence[float]) -> DeliveryPlan:
        visit_share = min(1.0, max(0.0, float(action[0])))
        weights = torch.tensor([float(a) for a in action[1:]], dtype=torch.float64)
        if bool((weights == 0).all()):
            return DeliveryPlan(visit_share=visit_share)
        features = _feature_matrix(self.sim.c, self.sim.regimen)
        # _cap serves the lowest score first, so a positive weight has to lower it.
        return DeliveryPlan(visit_share=visit_share, priority=-(weights @ features))

    def _reward(self, cost_before: float, qalys_before: float) -> float:
        cost = self.sim.ledger.total_discounted(self.categories) - cost_before
        qalys = self.sim.ledger.total_qalys() - qalys_before
        return (self.wtp * qalys - cost) * self.per_1000

    def _observe(self) -> torch.Tensor:
        sim = self.sim
        c = sim.c
        step = min(sim.step, sim.cfg.n_steps - 1)
        active = c.active(step)
        n = max(1, int(active.sum()))
        per_1000 = 1000.0 / n
        on_prog = active & c.on_program
        ever = active & c.ever_initiated
        groups = [float((active & (c.exposure_group == g)).sum()) / n
                  for g in range(N_EXPOSURE_GROUPS)]
        return torch.tensor([
            sim.step / sim.cfg.n_steps,
            n / max(1, sim.cfg.n_individuals),
            float(on_prog.sum()) / n,
            float((active & c.lapsed).sum()) / max(1, int(ever.sum())),
            float((active & (c.housing == int(Housing.UNSHELTERED))).sum()) / n,
            self._last["blocked"] * per_1000,
            self._last["contacts"] * per_1000,
            self._last["initiations"] * per_1000,
            self._last["diagnoses"] * per_1000,
            *groups,
        ], dtype=torch.float64)


# -- policies and their evaluation -------------------------------------------

def constant_policy(action: Sequence[float]) -> Callable[[torch.Tensor], Sequence[float]]:
    """The same action every week. Enough for any non-adaptive priority rule."""
    return lambda _obs: action


default_policy = constant_policy(DEFAULT_ACTION)
"""The model's own rationing: visits first, and random within each queue."""


@dataclass
class Evaluation:
    """A mean with a standard error, because a single episode is not a result."""

    label: str
    seeds: list[int]
    returns: list[float]
    infections: list[float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.returns)

    @property
    def se(self) -> float:
        if len(self.returns) < 2:
            return float("nan")
        return statistics.stdev(self.returns) / math.sqrt(len(self.returns))

    def __str__(self) -> str:
        return (f"{self.label}: net benefit per 1000 {self.mean:,.0f} "
                f"+/- {self.se:,.0f} over {len(self.seeds)} seeds")


def evaluate(env: PrepEnv, policy, seeds: Sequence[int], label: str = "policy") -> Evaluation:
    episodes = [env.run(policy, seed) for seed in seeds]
    return Evaluation(label=label, seeds=list(seeds),
                      returns=[e.ret for e in episodes],
                      infections=[e.infections for e in episodes])


@dataclass
class Difference:
    """A paired comparison. The pairing is the whole point.

    Both policies see the same seeds, so they see the same people, the same
    partnerships and the same acquisition draws. The difference per seed removes
    almost all of the variance that would otherwise swamp it, and the standard
    error reported here is the standard error of that difference, not of either
    policy on its own.
    """

    label: str
    per_seed: list[float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.per_seed)

    @property
    def se(self) -> float:
        if len(self.per_seed) < 2:
            return float("nan")
        return statistics.stdev(self.per_seed) / math.sqrt(len(self.per_seed))

    @property
    def z(self) -> float:
        return self.mean / self.se if self.se else float("nan")

    def is_an_improvement(self, standard_errors: float = 2.0) -> bool:
        return self.mean > standard_errors * self.se

    def __str__(self) -> str:
        verdict = ("better" if self.is_an_improvement() else
                   "not distinguishable from the reference")
        return (f"{self.label}: {self.mean:+,.0f} +/- {self.se:,.0f} net benefit per 1000 "
                f"({self.z:+.1f} se) -- {verdict}")


def compare(env: PrepEnv, policy, reference=default_policy,
            seeds: Sequence[int] = (), label: str = "policy vs default") -> Difference:
    per_seed = [env.run(policy, s).ret - env.run(reference, s).ret for s in seeds]
    return Difference(label=label, per_seed=per_seed)


def seed_bank(base: int, n: int, held_out: int = 0) -> tuple[list[int], list[int]]:
    """Training and held-out seeds, disjoint by construction.

    Tuning a policy on a seed bank and reporting its advantage on the same bank
    measures how well the policy fits those draws. The held-out list is what a
    reported number has to come from.
    """
    seeds = [base + 7919 * i for i in range(n + held_out)]
    return seeds[:n], seeds[n:]
