"""Comparisons between arms, and the uncertainty around them.

Three separate claims are kept separate throughout, because they are routinely
conflated and they are not the same question.

    health benefit        does the intervention avert infections and gain QALYs
    cost-effectiveness    is the cost per QALY below a stated threshold
    cost saving           does total spending fall

An intervention can pass the first two and fail the third. The comparison
objects below therefore carry all three, and ``cost_saving`` is a strict test on
the sign of incremental cost, never inferred from a favourable ICER.

Arms are paired. They share one starting population and one set of random
streams, so an incremental result is a difference between two histories of the
same people rather than a difference between two samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import simulate
from .config import Config
from .params import Params, Registry
from .rng import Stream, uniform
from .simulate import ArmResult

# An ICER is undefined when the intervention is dominant or dominated; those
# cases are reported by name instead of by a number that would mislead.
DOMINANT = "dominant"            # cheaper and better
DOMINATED = "dominated"          # costlier and worse


@dataclass
class Comparison:
    """One arm against the reference, under one perspective."""

    arm_id: str
    arm_name: str
    reference_id: str
    perspective: str
    delta_cost: float
    delta_qalys: float
    delta_life_years: float
    infections_averted: float
    icer: float | str
    nmb: dict[float, float]
    cost_saving: bool
    delta_cost_by_category: dict[str, float]

    def to_json(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "arm_name": self.arm_name,
            "reference_id": self.reference_id,
            "perspective": self.perspective,
            "delta_cost": self.delta_cost,
            "delta_qalys": self.delta_qalys,
            "delta_life_years": self.delta_life_years,
            "infections_averted": self.infections_averted,
            "icer": self.icer,
            "nmb": {str(k): v for k, v in self.nmb.items()},
            "cost_saving": self.cost_saving,
            "delta_cost_by_category": self.delta_cost_by_category,
        }


def icer(delta_cost: float, delta_qalys: float, qaly_tolerance: float = 1e-9) -> float | str:
    """Cost per QALY, or the name of the dominance relation when that is what holds.

    A negative ratio is ambiguous on its face: it arises both when an option is
    cheaper and better and when it is costlier and worse. Returning a string in
    those two cases makes the ambiguity impossible to print by accident.
    """
    if abs(delta_qalys) <= qaly_tolerance:
        return DOMINATED if delta_cost > 0 else DOMINANT
    if delta_qalys > 0 and delta_cost <= 0:
        return DOMINANT
    if delta_qalys < 0 and delta_cost >= 0:
        return DOMINATED
    if delta_qalys < 0 and delta_cost < 0:
        # Cheaper but worse. The ratio is the QALY price of disinvesting; it is
        # a real number but must be read against the threshold in reverse.
        return delta_cost / delta_qalys
    return delta_cost / delta_qalys


def compare(intervention: ArmResult, reference: ArmResult, cfg: Config,
            perspective: str) -> Comparison:
    cats = cfg.perspectives[perspective]
    d_cost = intervention.cost(cats) - reference.cost(cats)
    d_qaly = intervention.qalys - reference.qalys
    d_ly = intervention.epi["life_years_discounted"] - reference.epi["life_years_discounted"]
    inter_cat, ref_cat = intervention.ledger.by_category(), reference.ledger.by_category()
    return Comparison(
        arm_id=intervention.arm_id,
        arm_name=intervention.arm_name,
        reference_id=reference.arm_id,
        perspective=perspective,
        delta_cost=d_cost,
        delta_qalys=d_qaly,
        delta_life_years=d_ly,
        infections_averted=reference.epi["infections"] - intervention.epi["infections"],
        icer=icer(d_cost, d_qaly),
        nmb={w: w * d_qaly - d_cost for w in cfg.wtp},
        cost_saving=d_cost < 0,
        delta_cost_by_category={c: inter_cat[c] - ref_cat[c] for c in cats},
    )


def efficiency_frontier(results: dict[str, ArmResult], cfg: Config,
                        perspective: str) -> list[dict]:
    """The frontier, with simple and extended dominance removed explicitly.

    Reporting every arm against a common reference hides the case where an arm
    is beaten not by any single alternative but by a mixture of two. Extended
    dominance is removed here by the standard sweep: sort by cost, drop anything
    with fewer QALYs than a cheaper option, then drop anything whose incremental
    ratio exceeds that of the next option along.
    """
    cats = cfg.perspectives[perspective]
    points = sorted(
        ({"arm_id": r.arm_id, "arm_name": r.arm_name,
          "cost": r.cost(cats), "qalys": r.qalys, "status": "on_frontier"}
         for r in results.values()),
        key=lambda d: (d["cost"], -d["qalys"]),
    )

    kept: list[dict] = []
    best_qaly = -math.inf
    for pt in points:
        if pt["qalys"] <= best_qaly:
            pt["status"] = "dominated"
        else:
            best_qaly = pt["qalys"]
            kept.append(pt)

    changed = True
    while changed and len(kept) > 2:
        changed = False
        for i in range(1, len(kept) - 1):
            r_here = ((kept[i]["cost"] - kept[i - 1]["cost"]) /
                      (kept[i]["qalys"] - kept[i - 1]["qalys"]))
            r_next = ((kept[i + 1]["cost"] - kept[i]["cost"]) /
                      (kept[i + 1]["qalys"] - kept[i]["qalys"]))
            if r_here > r_next:
                kept[i]["status"] = "extended_dominated"
                kept.pop(i)
                changed = True
                break

    for i, pt in enumerate(kept):
        pt["icer_vs_previous"] = (
            None if i == 0 else
            icer(pt["cost"] - kept[i - 1]["cost"], pt["qalys"] - kept[i - 1]["qalys"])
        )
    return points


def break_even_price(intervention: ArmResult, reference: ArmResult, cfg: Config,
                     perspective: str, current_price: float,
                     wtp: float | None = None) -> dict[str, float | None]:
    """The acquisition price at which incremental cost, or net benefit, is zero.

    Total cost is affine in the price of a dose and nothing in the model reacts
    to the price, so the slope is the discounted dose count and no search is
    needed. ``cost_neutral_price`` is where the intervention stops costing more
    than usual care; ``threshold_price`` is where it stops being worth buying at
    ``wtp``. They are different numbers and both are reported.
    """
    cats = cfg.perspectives[perspective]
    slope = intervention.ledger.discounted_doses - reference.ledger.discounted_doses
    d_cost = intervention.cost(cats) - reference.cost(cats)
    if "program_drug" not in cats or slope <= 0:
        return {"cost_neutral_price": None, "threshold_price": None,
                "discounted_doses": slope, "current_price": current_price}
    # d_cost(price) = d_cost(current) + slope * (price - current)
    out: dict[str, float | None] = {
        "discounted_doses": slope,
        "current_price": current_price,
        "cost_neutral_price": current_price - d_cost / slope,
        "threshold_price": None,
    }
    if wtp is not None:
        nmb = wtp * (intervention.qalys - reference.qalys) - d_cost
        out["threshold_price"] = current_price + nmb / slope
    return out


@dataclass
class RunOutcome:
    """Everything one paired set of arm runs produced, ready to be tabulated."""

    label: str
    arms: dict[str, ArmResult]
    comparisons: dict[str, list[Comparison]]   # perspective -> comparisons
    frontier: dict[str, list[dict]]
    prices: dict[str, dict[str, float | None]]
    notes: list[str] = field(default_factory=list)
    replicate_traces: list[dict[str, list[float]]] = field(default_factory=list)
    """Cumulative infections per arm per replicate, for the uncertainty band."""

    def primary(self, cfg: Config) -> list[Comparison]:
        return self.comparisons[cfg.primary_perspective]

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "arms": {k: {"arm_name": r.arm_name, "epi": r.epi,
                         "cost_by_category": r.ledger.by_category(),
                         "qalys": r.qalys,
                         "counters": r.counters.__dict__,
                         "accounting": r.accounting,
                         "calibration_outputs": r.calibration_outputs}
                     for k, r in self.arms.items()},
            "comparisons": {p: [c.to_json() for c in cs]
                            for p, cs in self.comparisons.items()},
            "frontier": self.frontier,
            "prices": self.prices,
            "notes": self.notes,
            "replicate_traces": self.replicate_traces,
        }


def _mean_arms(per_replicate: list[dict[str, ArmResult]]) -> dict[str, ArmResult]:
    """Keep the first replicate's objects; replicate averaging happens at the
    comparison level, where the paired differences live."""
    return per_replicate[0]


def run_scenario(cfg: Config, p: Params, label: str = "base",
                 replicates: int | None = None,
                 frozen_prevalence: bool = False) -> RunOutcome:
    """One configuration, every arm, averaged over stochastic replicates.

    Replicates are averaged as paired differences rather than as separate means,
    because the pairing is the whole point of the common random numbers.
    """
    n_rep = replicates if replicates is not None else \
        int(cfg.section("uncertainty").get("stochastic_replicates", 1))
    per_rep = [simulate.run_all_arms(cfg, p, replicate=r, frozen_prevalence=frozen_prevalence)
               for r in range(n_rep)]
    ref_id = cfg.reference_arm

    comparisons: dict[str, list[Comparison]] = {}
    for persp in cfg.perspectives:
        by_arm: dict[str, list[Comparison]] = {}
        for res in per_rep:
            for arm_id, r in res.items():
                if arm_id == ref_id:
                    continue
                by_arm.setdefault(arm_id, []).append(compare(r, res[ref_id], cfg, persp))
        comparisons[persp] = [_average(cs, cfg) for cs in by_arm.values()]

    arms = _mean_arms(per_rep)
    frontier = {persp: efficiency_frontier(arms, cfg, persp) for persp in cfg.perspectives}

    prices: dict[str, dict[str, float | None]] = {}
    persp = cfg.primary_perspective
    for arm in cfg.arms:
        if not arm.injectable_uptake:
            continue
        price = p["cost_len_per_dose"] if arm.product == "lenacapavir" else p["cost_cab_per_dose"]
        prices[arm.id] = break_even_price(arms[arm.id], arms[ref_id], cfg, persp,
                                          price, wtp=cfg.wtp[1] if len(cfg.wtp) > 1 else None)

    notes = []
    if n_rep == 1:
        notes.append("One stochastic replicate: the Monte Carlo error on every difference "
                     "below is unmeasured.")
    return RunOutcome(label=label, arms=arms, comparisons=comparisons,
                      frontier=frontier, prices=prices, notes=notes,
                      replicate_traces=[{k: r.trace["cumulative_infections"]
                                         for k, r in res.items()} for res in per_rep])


def _average(cs: list[Comparison], cfg: Config) -> Comparison:
    """Average paired differences across replicates, then recompute the ratios."""
    n = len(cs)
    d_cost = sum(c.delta_cost for c in cs) / n
    d_qaly = sum(c.delta_qalys for c in cs) / n
    cats = list(cs[0].delta_cost_by_category)
    return Comparison(
        arm_id=cs[0].arm_id,
        arm_name=cs[0].arm_name,
        reference_id=cs[0].reference_id,
        perspective=cs[0].perspective,
        delta_cost=d_cost,
        delta_qalys=d_qaly,
        delta_life_years=sum(c.delta_life_years for c in cs) / n,
        infections_averted=sum(c.infections_averted for c in cs) / n,
        icer=icer(d_cost, d_qaly),
        nmb={w: w * d_qaly - d_cost for w in cfg.wtp},
        cost_saving=d_cost < 0,
        delta_cost_by_category={k: sum(c.delta_cost_by_category[k] for c in cs) / n
                                for k in cats},
    )


# -- one-way and two-way sensitivity -----------------------------------------

def _at_quantile(registry: Registry, cfg: Config, name: str, u: float) -> Params:
    return registry.resolve(cfg.price_year, overrides=cfg.overrides, draw={name: u})


def one_way(cfg: Config, registry: Registry, names: list[str], arm_id: str,
            replicates: int = 1) -> list[dict]:
    """Each parameter at its 2.5th and 97.5th percentile, everything else at base.

    Reported as the swing in incremental net monetary benefit, which is what a
    tornado diagram should rank by; ranking by ICER swing is unreadable whenever
    an arm crosses into dominance.
    """
    wtp = cfg.wtp[1] if len(cfg.wtp) > 1 else cfg.wtp[0]
    persp = cfg.primary_perspective
    base_p = registry.resolve(cfg.price_year, overrides=cfg.overrides)
    base = _nmb_for(cfg, base_p, arm_id, persp, wtp, replicates)

    rows = []
    for name in names:
        if name not in registry.rows:
            rows.append({"parameter": name, "error": "not in the evidence registry"})
            continue
        row = registry.rows[name]
        if row.distribution == "fixed":
            rows.append({"parameter": name, "error": "fixed: no range to vary over"})
            continue
        low = _nmb_for(cfg, _at_quantile(registry, cfg, name, 0.025), arm_id, persp,
                       wtp, replicates)
        high = _nmb_for(cfg, _at_quantile(registry, cfg, name, 0.975), arm_id, persp,
                        wtp, replicates)
        rows.append({
            "parameter": name,
            "low_value": row.quantile(0.025),
            "high_value": row.quantile(0.975),
            "base_nmb": base,
            "low_nmb": low,
            "high_nmb": high,
            "swing": abs(high - low),
        })
    rows.sort(key=lambda d: d.get("swing", -1.0), reverse=True)
    return rows


def _nmb_for(cfg: Config, p: Params, arm_id: str, perspective: str, wtp: float,
             replicates: int) -> float:
    out = run_scenario(cfg, p, label="sensitivity", replicates=replicates)
    for c in out.comparisons[perspective]:
        if c.arm_id == arm_id:
            return c.nmb[wtp] if wtp in c.nmb else wtp * c.delta_qalys - c.delta_cost
    raise KeyError(f"arm {arm_id!r} has no comparison against the reference")


def two_way(cfg: Config, registry: Registry, name_a: str, name_b: str, arm_id: str,
            n: int = 3, replicates: int = 1) -> list[dict]:
    """A grid over two parameters, reporting incremental cost as well as net benefit.

    The plan asks specifically for net cost across incidence and drug price,
    because that pair decides the cost-saving claim rather than the
    cost-effectiveness one.
    """
    for name in (name_a, name_b):
        if name not in registry.rows or registry.rows[name].distribution == "fixed":
            raise KeyError(f"two-way axis {name!r} is not a registry parameter with a range. "
                           "Name the parameter that is actually varied rather than a derived "
                           "target, so the figure's axis is what the model was given.")
    persp = cfg.primary_perspective
    wtp = cfg.wtp[1] if len(cfg.wtp) > 1 else cfg.wtp[0]
    qs = [0.025 + i * (0.95 / (n - 1)) for i in range(n)] if n > 1 else [0.5]
    grid = []
    for ua in qs:
        for ub in qs:
            p = registry.resolve(cfg.price_year, overrides=cfg.overrides,
                                 draw={name_a: ua, name_b: ub})
            out = run_scenario(cfg, p, label="two_way", replicates=replicates)
            c = next(c for c in out.comparisons[persp] if c.arm_id == arm_id)
            grid.append({
                name_a: p[name_a],
                name_b: p[name_b],
                "delta_cost": c.delta_cost,
                "delta_qalys": c.delta_qalys,
                "nmb": c.nmb[wtp] if wtp in c.nmb else wtp * c.delta_qalys - c.delta_cost,
                "cost_saving": c.cost_saving,
            })
    return grid


# -- probabilistic analysis ---------------------------------------------------

def psa_draws(cfg: Config, registry: Registry, n_draws: int,
              ensemble: list[dict[str, float]] | None = None) -> list[dict]:
    """``n_draws`` parameter sets, as (draw dict, override dict) pairs.

    Parameters the calibration identified are resampled from the accepted
    ensemble, not from their priors, because their priors no longer describe
    what is believed about them once the targets have been used. Everything else
    is drawn from the registry's own uncertainty distribution.
    """
    calibrated = set(ensemble[0]) if ensemble else set()
    varying = [n for n, r in registry.rows.items()
               if r.distribution != "fixed" and n not in calibrated
               and n not in cfg.overrides]
    idx = torch.arange(n_draws, dtype=torch.int64)
    us = {name: uniform(cfg.seed, Stream.ATTR_HIV, 31_000 + i, idx)
          for i, name in enumerate(varying)}
    pick = uniform(cfg.seed, Stream.ATTR_CARE, 31_999, idx)

    sets = []
    for j in range(n_draws):
        draw = {name: float(us[name][j]) for name in varying}
        overrides = dict(cfg.overrides)
        if ensemble:
            member = ensemble[int(float(pick[j]) * len(ensemble)) % len(ensemble)]
            overrides.update(member)
        sets.append({"draw": draw, "overrides": overrides})
    return sets


@dataclass
class PSAResult:
    arm_id: str
    arm_name: str
    perspective: str
    n_draws: int
    delta_cost: list[float]
    delta_qalys: list[float]
    infections_averted: list[float]
    nmb: dict[float, list[float]]

    def summary(self) -> dict:
        def band(x: list[float]) -> dict[str, float]:
            t = torch.tensor(x, dtype=torch.float64)
            return {"mean": float(t.mean()), "median": float(t.median()),
                    "p2_5": float(t.quantile(0.025)), "p97_5": float(t.quantile(0.975))}
        n = max(1, self.n_draws)
        return {
            "arm_id": self.arm_id,
            "arm_name": self.arm_name,
            "perspective": self.perspective,
            "n_draws": self.n_draws,
            "delta_cost": band(self.delta_cost),
            "delta_qalys": band(self.delta_qalys),
            "infections_averted": band(self.infections_averted),
            "probability_cost_saving": sum(1 for d in self.delta_cost if d < 0) / n,
            "probability_qaly_gain": sum(1 for d in self.delta_qalys if d > 0) / n,
            "probability_cost_effective": {
                str(w): sum(1 for v in vals if v > 0) / n for w, vals in self.nmb.items()
            },
            "nmb": {str(w): band(v) for w, v in self.nmb.items()},
        }


def run_psa(cfg: Config, registry: Registry, n_draws: int,
            ensemble: list[dict[str, float]] | None = None,
            replicates: int = 1) -> dict[str, PSAResult]:
    persp = cfg.primary_perspective
    sets = psa_draws(cfg, registry, n_draws, ensemble)
    acc: dict[str, PSAResult] = {}
    for s in sets:
        p = registry.resolve(cfg.price_year, overrides=s["overrides"], draw=s["draw"])
        out = run_scenario(cfg, p, label="psa", replicates=replicates)
        for c in out.comparisons[persp]:
            r = acc.setdefault(c.arm_id, PSAResult(
                arm_id=c.arm_id, arm_name=c.arm_name, perspective=persp, n_draws=0,
                delta_cost=[], delta_qalys=[], infections_averted=[],
                nmb={w: [] for w in cfg.wtp}))
            r.n_draws += 1
            r.delta_cost.append(c.delta_cost)
            r.delta_qalys.append(c.delta_qalys)
            r.infections_averted.append(c.infections_averted)
            for w in cfg.wtp:
                r.nmb[w].append(c.nmb[w])
    return acc


def ceac(psa: PSAResult, thresholds: list[float]) -> list[dict[str, float]]:
    """Probability the arm is cost-effective as the threshold moves.

    Computed from the stored incremental cost and QALY pairs rather than from
    the stored NMB, so the curve can be drawn at thresholds the run did not use.
    """
    n = max(1, psa.n_draws)
    out = []
    for w in thresholds:
        wins = sum(1 for dc, dq in zip(psa.delta_cost, psa.delta_qalys) if w * dq - dc > 0)
        out.append({"threshold": w, "probability": wins / n})
    return out


# -- structural sensitivity ---------------------------------------------------

def structural_scenarios(cfg: Config, registry: Registry,
                         replicates: int = 1) -> dict[str, RunOutcome]:
    """The scenarios that change what the model assumes, not what it is given.

    ``direct_effects_only`` is not a parameter change at all: it freezes
    infectious prevalence at its baseline, which removes herd effects and gives
    the conservative bound on the intervention's benefit.
    """
    out: dict[str, RunOutcome] = {}
    for name, spec in cfg.section("experiments").get("structural", {}).items():
        if spec is True:
            p = registry.resolve(cfg.price_year, overrides=cfg.overrides)
            out[name] = run_scenario(cfg, p, label=name, replicates=replicates,
                                     frozen_prevalence=True)
            continue
        if not isinstance(spec, dict):
            continue
        overrides = dict(cfg.overrides)
        overrides.update({k: float(v) for k, v in spec.items()})
        p = registry.resolve(cfg.price_year, overrides=overrides)
        out[name] = run_scenario(cfg, p, label=name, replicates=replicates)
    return out


def scenario_grid(cfg: Config, registry: Registry, arm_id: str,
                  replicates: int = 1) -> list[dict]:
    """The plan's illustrative grid: one axis at a time, over stated values.

    A full cross-product of the five axes is thousands of microsimulations and
    would say little the one-way results do not; each axis is swept with the
    others at base, and that limitation is stated rather than hidden.
    """
    spec = cfg.section("experiments").get("scenario_grid") or {}
    axis_map = {
        "baseline_annual_risk": "lambda_sex_scale",
        "program_coverage": "accept_prob_lowbarrier",
        "visit_retention": "attendance_prob_lowbarrier",
        "drug_cost_per_year": None,      # handled as price per dose below
        "delivery_cost_per_year": None,
    }
    persp = cfg.primary_perspective
    wtp = cfg.wtp[1] if len(cfg.wtp) > 1 else cfg.wtp[0]
    base_p = registry.resolve(cfg.price_year, overrides=cfg.overrides)
    rows = []
    for axis, values in spec.items():
        for v in values:
            overrides = dict(cfg.overrides)
            if axis == "drug_cost_per_year":
                overrides["cost_len_per_dose"] = float(v) / 2.0
                overrides["cost_cab_per_dose"] = float(v) / 6.0
            elif axis == "delivery_cost_per_year":
                # An annual delivery budget per participant spread over contacts.
                overrides["cost_contact_lowbarrier"] = float(v) / 12.0
                overrides["cost_contact_clinic"] = float(v) / 12.0
            elif axis == "baseline_annual_risk":
                scale = float(v) / max(1e-12, _implied_annual_risk(base_p))
                overrides["lambda_sex_scale"] = base_p["lambda_sex_scale"] * scale
            else:
                overrides[axis_map[axis]] = float(v)
            p = registry.resolve(cfg.price_year, overrides=overrides)
            out = run_scenario(cfg, p, label=f"{axis}={v}", replicates=replicates)
            c = next(c for c in out.comparisons[persp] if c.arm_id == arm_id)
            rows.append({"axis": axis, "value": v, "delta_cost": c.delta_cost,
                         "delta_qalys": c.delta_qalys, "icer": c.icer,
                         "nmb": c.nmb.get(wtp), "cost_saving": c.cost_saving,
                         "infections_averted": c.infections_averted})
    return rows


def _implied_annual_risk(p: Params) -> float:
    """A rough annual acquisition hazard implied by the current scale parameters.

    Only used to rescale the transmission parameter onto the plan's stated risk
    axis. It is an approximation and the reported axis value is the target, not
    a measured model output.
    """
    return p["lambda_sex_scale"] * p["baseline_hiv_prevalence"] + p["external_hazard_sex_annual"]
