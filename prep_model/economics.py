"""Resource use, payer ledgers, QALYs and discounting.

Costs are kept in separate category ledgers and a perspective is a set of
categories, so the programme-operator question ("can we fund delivery?") and
the consolidated-public-sector question ("does total public spending fall?")
are answered from the same run without one being silently substituted for the
other. A payment is counted once, in one category.

Two habits this module is built to prevent. Adding a single lifetime HIV cost
on top of simulated care costs, which double counts: the terminal value here
covers only time strictly after the horizon and uses the same state-specific
annual costs the simulation used. And treating improved survival as free:
people who live longer keep accruing non-HIV care and housing costs, and those
are included rather than dropped to make the intervention look better.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .params import Params
from .population import Cohort
from .states import CATEGORY_INDEX, COST_CATEGORIES, CareState, HivStage, Housing

DAYS_PER_YEAR = 365.25


def annual_hiv_care_cost(stage: torch.Tensor, care: torch.Tensor, p: Params) -> torch.Tensor:
    """HIV-attributable annual medical cost. Zero for susceptible people."""
    cost = torch.zeros(stage.shape, dtype=torch.float64)
    with_hiv = stage != int(HivStage.SUSCEPTIBLE)
    by_care = torch.zeros_like(cost)
    by_care = torch.where(care == int(CareState.UNDIAGNOSED),
                          torch.full_like(cost, p["cost_hiv_care_undiagnosed"]), by_care)
    by_care = torch.where(care == int(CareState.DIAGNOSED_NO_ART),
                          torch.full_like(cost, p["cost_hiv_care_diagnosed_noart"]), by_care)
    by_care = torch.where(care == int(CareState.ART_UNSUPPRESSED),
                          torch.full_like(cost, p["cost_hiv_care_art_unsuppressed"]), by_care)
    by_care = torch.where(care == int(CareState.ART_SUPPRESSED),
                          torch.full_like(cost, p["cost_hiv_care_art_suppressed"]), by_care)
    advanced = torch.where(stage == int(HivStage.ADVANCED),
                           torch.full_like(cost, p["cost_hiv_care_advanced"]), cost)
    return torch.where(with_hiv, by_care + advanced, cost)


def annual_housing_cost(housing: torch.Tensor, stage: torch.Tensor, p: Params) -> torch.Tensor:
    """Housing-system cost per person-year, plus any HIV-attributable increment.

    The increment is zero in the primary analysis. Ordinary shelter expenditure
    is a background service that appears in every arm; a prevented infection
    does not mean a person stops needing a bed.
    """
    per_day = torch.zeros(housing.shape, dtype=torch.float64)
    per_day = torch.where(housing == int(Housing.UNSHELTERED),
                          torch.full_like(per_day, p["cost_housing_day_unsheltered"]), per_day)
    per_day = torch.where(housing == int(Housing.SHELTERED),
                          torch.full_like(per_day, p["cost_housing_day_sheltered"]), per_day)
    per_day = torch.where(housing == int(Housing.HOUSED),
                          torch.full_like(per_day, p["cost_housing_day_housed"]), per_day)
    extra = torch.where(stage != int(HivStage.SUSCEPTIBLE),
                        torch.full_like(per_day, p["cost_hiv_attributable_housing_annual"]),
                        torch.zeros_like(per_day))
    return per_day * DAYS_PER_YEAR + extra


def utility(stage: torch.Tensor, care: torch.Tensor, housing: torch.Tensor,
            p: Params) -> torch.Tensor:
    """Health utility per year lived, after the housing decrement."""
    u = torch.full(stage.shape, p["utility_hiv_negative"], dtype=torch.float64)
    with_hiv = stage != int(HivStage.SUSCEPTIBLE)
    hiv_u = torch.where(care == int(CareState.UNDIAGNOSED),
                        torch.full_like(u, p["utility_hiv_undiagnosed"]),
                        torch.full_like(u, p["utility_hiv_diagnosed_noart"]))
    hiv_u = torch.where(care >= int(CareState.ART_UNSUPPRESSED),
                        torch.full_like(u, p["utility_hiv_art"]), hiv_u)
    hiv_u = torch.where(stage == int(HivStage.ADVANCED),
                        torch.full_like(u, p["utility_hiv_advanced"]), hiv_u)
    u = torch.where(with_hiv, hiv_u, u)
    u = torch.where(housing == int(Housing.UNSHELTERED), u - p["utility_decrement_unsheltered"], u)
    u = torch.where(housing == int(Housing.SHELTERED), u - p["utility_decrement_sheltered"], u)
    return u.clamp_min(0.0)


@dataclass
class Ledger:
    """Discounted and undiscounted cost by category and by year, plus QALYs."""

    n_steps: int
    dt: float
    discount_rate: float
    n_years: int
    discounted: torch.Tensor = field(init=False)
    undiscounted_by_year: torch.Tensor = field(init=False)
    qalys: float = 0.0
    life_years: float = 0.0
    terminal_discounted: torch.Tensor = field(init=False)
    terminal_qalys: float = 0.0
    discounted_doses: float = 0.0
    """Injectable doses, discounted the same way their cost is.

    Total cost is affine in the acquisition price, so this is the slope and the
    break-even price can be solved from a single run rather than searched for.
    """

    def __post_init__(self) -> None:
        k = len(COST_CATEGORIES)
        self.discounted = torch.zeros(k, dtype=torch.float64)
        self.undiscounted_by_year = torch.zeros(k, self.n_years, dtype=torch.float64)
        self.terminal_discounted = torch.zeros(k, dtype=torch.float64)

    def factor(self, step: int) -> float:
        return (1.0 + self.discount_rate) ** (-step * self.dt)

    def accrue(self, category: str, amount: torch.Tensor | float, step: int) -> None:
        total = float(amount.sum()) if torch.is_tensor(amount) else float(amount)
        if total == 0.0:
            return
        i = CATEGORY_INDEX[category]
        self.discounted[i] += total * self.factor(step)
        year = min(int(step * self.dt), self.n_years - 1)
        self.undiscounted_by_year[i, year] += total

    def accrue_terminal(self, category: str, amount: torch.Tensor | float) -> None:
        total = float(amount.sum()) if torch.is_tensor(amount) else float(amount)
        self.terminal_discounted[CATEGORY_INDEX[category]] += total

    def total_discounted(self, categories: list[str]) -> float:
        return float(sum(self.discounted[CATEGORY_INDEX[c]] +
                         self.terminal_discounted[CATEGORY_INDEX[c]] for c in categories))

    def by_category(self) -> dict[str, float]:
        return {c: float(self.discounted[CATEGORY_INDEX[c]] +
                         self.terminal_discounted[CATEGORY_INDEX[c]])
                for c in COST_CATEGORIES}

    def annual_table(self) -> dict[str, list[float]]:
        return {c: [float(x) for x in self.undiscounted_by_year[CATEGORY_INDEX[c]]]
                for c in COST_CATEGORIES}

    def total_qalys(self) -> float:
        return self.qalys + self.terminal_qalys


def accrue_step(ledger: Ledger, c: Cohort, active: torch.Tensor, p: Params,
                step: int, dt: float) -> None:
    """Background health-state costs and health for one step."""
    a = active
    ledger.accrue("hiv_care", annual_hiv_care_cost(c.hiv_stage[a], c.care_state[a], p) * dt, step)
    ledger.accrue("non_hiv_care", torch.full((int(a.sum()),), p["cost_non_hiv_care_annual"],
                                             dtype=torch.float64) * dt, step)
    ledger.accrue("housing", annual_housing_cost(c.housing[a], c.hiv_stage[a], p) * dt, step)

    u = utility(c.hiv_stage[a], c.care_state[a], c.housing[a], p)
    ledger.qalys += float(u.sum()) * dt * ledger.factor(step)
    ledger.life_years += float(a.sum()) * dt * ledger.factor(step)


def _remaining_life_expectancy(c: Cohort, active: torch.Tensor, p: Params) -> torch.Tensor:
    """Remaining years of life by age and HIV state, for the terminal value only."""
    neg = p["life_expectancy_hiv_negative_at_45"]
    sup = p["life_expectancy_hiv_suppressed_at_45"]
    unsup = p["life_expectancy_hiv_unsuppressed_at_45"]
    base = torch.full(c.age.shape, neg, dtype=torch.float64)
    with_hiv = c.hiv_stage != int(HivStage.SUSCEPTIBLE)
    base = torch.where(with_hiv, torch.full_like(base, unsup), base)
    base = torch.where(with_hiv & (c.care_state == int(CareState.ART_SUPPRESSED)),
                       torch.full_like(base, sup), base)
    # Linear age adjustment around the reference age of 45.
    adjusted = base + (45.0 - c.age) * 0.7
    return adjusted.clamp_min(1.0)[active]


def accrue_terminal(ledger: Ledger, c: Cohort, active: torch.Tensor, p: Params,
                    horizon_step: int) -> dict[str, float]:
    """Discounted consequences of the time strictly beyond the simulated horizon.

    An exponential-survival annuity: with a constant mortality hazard ``m`` and
    a continuous discount rate ``rho``, a constant stream of value ``v`` per
    year is worth ``v / (rho + m)``. Every quantity is then discounted back from
    the horizon. Nothing simulated before the horizon is repeated here.
    """
    a = active
    rho = math.log(1.0 + ledger.discount_rate)
    m = 1.0 / _remaining_life_expectancy(c, a, p)
    annuity = 1.0 / (rho + m)
    df = ledger.factor(horizon_step)

    ledger.accrue_terminal("hiv_care",
                           annual_hiv_care_cost(c.hiv_stage[a], c.care_state[a], p) * annuity * df)
    ledger.accrue_terminal("non_hiv_care", p["cost_non_hiv_care_annual"] * annuity.sum() * df)
    ledger.accrue_terminal("housing",
                           annual_housing_cost(c.housing[a], c.hiv_stage[a], p) * annuity * df)
    u = utility(c.hiv_stage[a], c.care_state[a], c.housing[a], p)
    ledger.terminal_qalys += float((u * annuity).sum()) * df
    return {"terminal_people": float(a.sum()), "terminal_mean_annuity": float(annuity.mean())}


def present_value_constant_stream(annual: float, years: float, rate: float) -> float:
    """Analytic check used by the tests: discrete annual discounting."""
    if rate == 0:
        return annual * years
    return annual * (1 - (1 + rate) ** (-years)) / rate
