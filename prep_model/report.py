"""Tables, figures and a generated report.

Every number printed here is read from a run output. Nothing is retyped, and
nothing is rounded into the prose by hand, because a retyped number is the most
common way a report stops matching its own model.

Two habits are enforced rather than requested. Inputs flagged ``hypothetical``
are listed automatically wherever they touched a result, so no reader has to
take a footnote's word for what was measured. And deaths are labelled as
model-estimated mortality outcomes, never as "lives saved" per infection
averted, which would ignore both modern HIV survival and competing causes.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import Config
from .experiments import DOMINANT, DOMINATED, PSAResult, RunOutcome, ceac
from .params import Params, Registry
from .states import COST_CATEGORIES

MPL_BACKEND = "Agg"


def _years(p: Params, cfg: Config, with_migration: bool) -> float:
    """Discounted years over which a treated infection actually accrues cost here.

    Leaving the modelled population ends the cost stream exactly as death does, so
    out-migration belongs in the hazard. Quoting the two side by side is the only
    way a reader can see how much of an infection's lifetime cost this model is in
    a position to count.
    """
    import math

    hazard = 1.0 / p["life_expectancy_hiv_suppressed_at_45"]
    if with_migration:
        hazard += p["migration_rate_per_year"]
    return 1.0 / (math.log(1.0 + cfg.discount_rate) + hazard)


# -- small formatting helpers -------------------------------------------------

def money(x: float | None) -> str:
    if x is None:
        return "n/a"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def num(x: float | None, dp: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{dp}f}"


def icer_str(v: float | str) -> str:
    if v == DOMINANT:
        return "dominant (cheaper and better)"
    if v == DOMINATED:
        return "dominated (costlier and worse)"
    return money(float(v)) + " per QALY"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# -- the seven required tables ------------------------------------------------

def table_inputs(registry: Registry, p: Params) -> str:
    """Table 1: input values, sources, uncertainty and assumptions."""
    rows = []
    for name in sorted(p.accessed):
        r = registry.rows[name]
        rng = ("fixed" if r.distribution == "fixed"
               else f"{r.distribution} [{num(r.lower, 4)}, {num(r.upper, 4)}]")
        rows.append([name, r.definition[:90], num(p[name], 4), r.units or "-",
                     r.currency_year or "-", rng, r.evidence_grade or "-",
                     r.assumption_flag, r.source_url or "none"])
    return table(["parameter", "definition", "value used", "units", "price year",
                  "uncertainty", "grade", "flag", "source"], rows)


def table_delivery(out: RunOutcome, cfg: Config) -> str:
    """Table 2: what the programme actually did, per arm."""
    rows = []
    for arm in cfg.arms:
        r = out.arms[arm.id]
        k = r.counters
        rows.append([
            arm.id, arm.name,
            f"{k.contacts:,}", f"{k.offers:,}", f"{k.acceptances:,}",
            f"{k.initiations:,}", f"{k.doses:,}", f"{k.missed_visits:,}",
            f"{k.reengagements:,}", f"{k.discontinuations:,}",
            num(r.epi["person_years_on_program"], 0),
            f"{k.tests:,}", f"{k.program_diagnoses:,}", f"{k.linked_to_art:,}",
            f"{k.capacity_blocked_initiations:,}",
        ])
    return table(["arm", "name", "contacts", "offers", "accepted", "initiations",
                  "injections", "missed visits", "re-engagements", "discontinuations",
                  "participant-years", "tests", "programme diagnoses", "linked to ART",
                  "initiations blocked by capacity"], rows)


def table_health(out: RunOutcome, cfg: Config, psa: dict[str, PSAResult] | None = None) -> str:
    """Table 3: health outcomes with uncertainty where the PSA supplies it."""
    ref = out.arms[cfg.reference_arm]
    rows = []
    for arm in cfg.arms:
        r = out.arms[arm.id]
        band = ""
        if psa and arm.id in psa:
            t = torch.tensor(psa[arm.id].infections_averted, dtype=torch.float64)
            band = f"{float(t.quantile(0.025)):,.0f} to {float(t.quantile(0.975)):,.0f}"
        rows.append([
            arm.id, arm.name,
            num(r.epi["infections"], 0),
            num(ref.epi["infections"] - r.epi["infections"], 0),
            band or "not sampled",
            num(r.epi["hiv_deaths"], 0),
            num(ref.epi["hiv_deaths"] - r.epi["hiv_deaths"], 0),
            num(r.epi["deaths"], 0),
            num(r.epi["life_years_discounted"], 0),
            num(r.qalys, 1),
            num(r.qalys - ref.qalys, 1),
        ])
    return table(["arm", "name", "infections", "infections averted",
                  "95% interval on infections averted", "HIV deaths",
                  "HIV deaths averted", "all-cause deaths", "discounted life-years",
                  "discounted QALYs", "incremental QALYs"], rows)


def table_annual_spending(out: RunOutcome, cfg: Config) -> str:
    """Table 4: undiscounted annual cash spending by category, years 1..budget_years."""
    n = cfg.budget_years
    payers = cfg.payers
    rows = []
    for arm in cfg.arms:
        annual = out.arms[arm.id].ledger.annual_table()
        for cat in COST_CATEGORIES:
            series = annual[cat][:n]
            if not any(series):
                continue
            rows.append([arm.id, cat, payers[cat]] +
                        [money(v) for v in series] + [money(sum(series))])
    return table(["arm", "cost category", "payer"] +
                 [f"year {i + 1}" for i in range(n)] + [f"years 1-{n}"], rows)


def table_lifetime_costs(out: RunOutcome, cfg: Config) -> str:
    """Table 5: discounted cost by category and arm, under the primary perspective."""
    cats = cfg.perspectives[cfg.primary_perspective]
    rows = []
    for arm in cfg.arms:
        by_cat = out.arms[arm.id].ledger.by_category()
        rows.append([arm.id, arm.name] + [money(by_cat[c]) for c in cats] +
                    [money(sum(by_cat[c] for c in cats))])
    return table(["arm", "name"] + cats + ["total"], rows)


def table_cost_effectiveness(out: RunOutcome, cfg: Config,
                             psa: dict[str, PSAResult] | None = None) -> str:
    """Table 6: incremental results, dominance, net benefit, probability of saving."""
    persp = cfg.primary_perspective
    status = {d["arm_id"]: d["status"] for d in out.frontier[persp]}
    rows = []
    for c in out.comparisons[persp]:
        p_save = "not sampled"
        if psa and c.arm_id in psa:
            s = psa[c.arm_id].summary()
            p_save = f"{s['probability_cost_saving']:.0%}"
        rows.append([c.arm_id, c.arm_name, money(c.delta_cost), num(c.delta_qalys, 1),
                     icer_str(c.icer), status.get(c.arm_id, "-")] +
                    [money(c.nmb[w]) for w in cfg.wtp] +
                    ["yes" if c.cost_saving else "no", p_save])
    return table(["arm", "name", "incremental cost", "incremental QALYs", "ICER",
                  "frontier status"] + [f"NMB at {money(w)}" for w in cfg.wtp] +
                 ["cost saving in base case", "probability of cost saving"], rows)


def table_price_thresholds(out: RunOutcome, cfg: Config,
                           grid: list[dict] | None = None) -> str:
    """Table 7: break-even acquisition price, and the retention the result assumes."""
    rows = []
    for arm_id, d in out.prices.items():
        r = out.arms[arm_id]
        attended = r.counters.doses
        scheduled = attended + r.counters.missed_visits
        retention = attended / scheduled if scheduled else float("nan")
        rows.append([
            arm_id, r.arm_name,
            money(d["current_price"]),
            money(d["cost_neutral_price"]),
            money(d["threshold_price"]),
            num(d["discounted_doses"], 0),
            f"{retention:.1%}",
        ])
    out_md = table(["arm", "name", "price per injection used",
                    "price at which incremental cost is zero",
                    "price at which net benefit is zero",
                    "discounted injections", "on-time visit share achieved"], rows)
    if grid:
        sub = [g for g in grid if g["axis"] == "baseline_annual_risk"]
        if sub:
            out_md += "\n\nBy baseline acquisition risk:\n\n" + table(
                ["target annual risk", "incremental cost", "incremental QALYs",
                 "infections averted", "cost saving"],
                [[num(g["value"], 4), money(g["delta_cost"]), num(g["delta_qalys"], 1),
                  num(g["infections_averted"], 0), "yes" if g["cost_saving"] else "no"]
                 for g in sub])
    return out_md


# -- the five required figures ------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use(MPL_BACKEND)
    import matplotlib.pyplot as plt
    return plt


def figure_cumulative_infections(out: RunOutcome, cfg: Config, path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    years = [i * cfg.dt for i in range(cfg.n_steps)]
    reps = out.replicate_traces
    for arm in cfg.arms:
        ax.plot(years, out.arms[arm.id].trace["cumulative_infections"],
                label=f"{arm.id} {arm.name}")
        if len(reps) > 1:
            band = [[rep[arm.id][s] for rep in reps] for s in range(cfg.n_steps)]
            ax.fill_between(years, [min(b) for b in band], [max(b) for b in band], alpha=0.15)
    ax.set_xlabel("year")
    ax.set_ylabel("cumulative HIV infections")
    ax.set_title("Cumulative infections by policy\n(bands are the range over stochastic replicates)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_annual_spending(out: RunOutcome, cfg: Config, path: Path) -> Path:
    """Annual cash, kept visually separate from the discounted lifetime total.

    The two are on separate axes on purpose: a present-value offset that arrives
    over decades is not money a budget holder has next year.
    """
    plt = _plt()
    n = cfg.budget_years
    arm_ids = [a.id for a in cfg.arms]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.8 / max(1, len(arm_ids))
    for j, arm_id in enumerate(arm_ids):
        annual = out.arms[arm_id].ledger.annual_table()
        bottom = [0.0] * n
        for cat in cfg.perspectives[cfg.primary_perspective]:
            series = annual[cat][:n]
            xs = [i + j * width for i in range(n)]
            ax1.bar(xs, series, width=width, bottom=bottom, label=cat if j == 0 else None)
            bottom = [b + s for b, s in zip(bottom, series)]
    ax1.set_xticks([i + 0.4 for i in range(n)])
    ax1.set_xticklabels([f"y{i + 1}" for i in range(n)])
    ax1.set_ylabel("undiscounted cash spending")
    ax1.set_title(f"Annual cash, years 1-{n}\n(bars grouped by arm: {', '.join(arm_ids)})")
    ax1.legend(fontsize=7)

    persp = cfg.primary_perspective
    deltas = [c.delta_cost for c in out.comparisons[persp]]
    ax2.bar([c.arm_id for c in out.comparisons[persp]], deltas,
            color=["tab:red" if d > 0 else "tab:green" for d in deltas])
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel(f"discounted lifetime incremental cost vs {cfg.reference_arm}")
    ax2.set_title("Lifetime present value\n(a different quantity from the cash on the left)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_ce_plane_and_ceac(out: RunOutcome, cfg: Config, path: Path,
                             psa: dict[str, PSAResult] | None = None) -> Path:
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    persp = cfg.primary_perspective
    for c in out.comparisons[persp]:
        if psa and c.arm_id in psa:
            r = psa[c.arm_id]
            ax1.scatter(r.delta_qalys, r.delta_cost, s=6, alpha=0.35, label=c.arm_id)
        ax1.scatter([c.delta_qalys], [c.delta_cost], marker="x", s=90, color="black")
        ax1.annotate(c.arm_id, (c.delta_qalys, c.delta_cost), fontsize=8)
    lim = ax1.get_xlim()
    for w in cfg.wtp:
        ax1.plot(lim, [w * lim[0], w * lim[1]], lw=0.7, ls="--",
                 label=f"{money(w)}/QALY")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_xlabel("incremental QALYs")
    ax1.set_ylabel("incremental cost")
    ax1.set_title(f"Cost-effectiveness plane ({persp})")
    ax1.legend(fontsize=7)

    if psa:
        thresholds = [i * 10_000 for i in range(0, 31)]
        for arm_id, r in psa.items():
            pts = ceac(r, thresholds)
            ax2.plot([d["threshold"] for d in pts], [d["probability"] for d in pts],
                     label=arm_id)
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=7)
    else:
        ax2.text(0.5, 0.5, "no probabilistic draws in this run",
                 ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("willingness to pay per QALY")
    ax2.set_ylabel("probability cost-effective")
    ax2.set_title("Acceptability curve")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_net_cost_heatmap(grid: list[dict], name_a: str, name_b: str, path: Path) -> Path:
    plt = _plt()
    xs = sorted({g[name_a] for g in grid})
    ys = sorted({g[name_b] for g in grid})
    z = [[next(g["delta_cost"] for g in grid if g[name_a] == x and g[name_b] == y)
          for x in xs] for y in ys]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    peak = max(abs(v) for row in z for v in row) or 1.0
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="RdYlGn_r",
                   vmin=-peak, vmax=peak)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:.3g}" for x in xs], rotation=45, ha="right")
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{y:.4g}" for y in ys])
    ax.set_xlabel(name_a)
    ax.set_ylabel(name_b)
    ax.set_title("Incremental cost\n(green is cost saving, red is added spending)")
    fig.colorbar(im, ax=ax, label="incremental cost")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_tornado(rows: list[dict], path: Path, wtp: float) -> Path:
    plt = _plt()
    rows = [r for r in rows if "swing" in r][:14][::-1]
    if not rows:
        return path
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(rows) + 2))
    base = rows[0]["base_nmb"]
    for i, r in enumerate(rows):
        lo, hi = sorted((r["low_nmb"], r["high_nmb"]))
        ax.barh(i, hi - lo, left=lo, color="tab:blue", alpha=0.7)
    ax.axvline(base, color="black", lw=1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["parameter"] for r in rows], fontsize=8)
    ax.set_xlabel(f"incremental net monetary benefit at {money(wtp)} per QALY")
    ax.set_title("What moves the decision\n(each input at its 2.5th and 97.5th percentile)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# -- provenance ---------------------------------------------------------------

@dataclass
class Provenance:
    code_revision: str
    seed: int
    config_path: str
    parameter_file_hash: str
    torch_version: str
    python_version: str
    platform: str
    device: str

    def to_json(self) -> dict:
        return self.__dict__


def provenance(cfg: Config) -> Provenance:
    import hashlib
    from .params import DATA_DIR
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=Path(__file__).parent.parent,
                             timeout=10).stdout.strip() or "not a git checkout"
    except (OSError, subprocess.SubprocessError):
        rev = "not a git checkout"
    digest = hashlib.sha256((DATA_DIR / "parameters.csv").read_bytes()).hexdigest()[:16]
    return Provenance(code_revision=rev, seed=cfg.seed, config_path=str(cfg.path),
                      parameter_file_hash=digest, torch_version=torch.__version__,
                      python_version=platform.python_version(),
                      platform=platform.platform(), device=cfg.device)


# -- the report ---------------------------------------------------------------

def conclusion(out: RunOutcome, cfg: Config, arm_id: str) -> str:
    """The plan's conclusion sentence, with the numbers filled from the run."""
    persp = cfg.primary_perspective
    c = next(c for c in out.comparisons[persp] if c.arm_id == arm_id)
    price = out.prices.get(arm_id, {})
    neutral = price.get("cost_neutral_price")
    per_year = None if neutral is None else neutral * (2 if "lenacapavir" in c.arm_name else 6)
    tail = ("it does not become cost saving at any non-negative acquisition price"
            if per_year is None or per_year <= 0 else
            f"it becomes cost saving below an annual acquisition price of {money(per_year)}")
    return (f"Under the specified epidemiological and delivery assumptions, arm {arm_id} "
            f"({c.arm_name}) is projected to avert {c.infections_averted:,.0f} infections "
            f"and gain {c.delta_qalys:,.1f} QALYs, with {money(c.delta_cost)} incremental "
            f"public expenditure over the {cfg.horizon_years:.0f}-year horizon plus its "
            f"terminal continuation; {tail}.")


def build_report(cfg: Config, registry: Registry, p: Params, out: RunOutcome,
                 out_dir: Path, calibration: dict | None = None,
                 psa: dict[str, PSAResult] | None = None,
                 one_way_rows: list[dict] | None = None,
                 two_way_grid: list[dict] | None = None,
                 two_way_names: tuple[str, str] | None = None,
                 grid_rows: list[dict] | None = None,
                 structural: dict[str, RunOutcome] | None = None) -> Path:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance(cfg)
    persp = cfg.primary_perspective
    wtp = cfg.wtp[1] if len(cfg.wtp) > 1 else cfg.wtp[0]

    figure_cumulative_infections(out, cfg, fig_dir / "fig1_cumulative_infections.png")
    figure_annual_spending(out, cfg, fig_dir / "fig2_annual_spending.png")
    figure_ce_plane_and_ceac(out, cfg, fig_dir / "fig3_ce_plane_ceac.png", psa)
    if two_way_grid and two_way_names:
        figure_net_cost_heatmap(two_way_grid, two_way_names[0], two_way_names[1],
                                fig_dir / "fig4_net_cost_heatmap.png")
    if one_way_rows:
        figure_tornado(one_way_rows, fig_dir / "fig5_tornado.png", wtp)

    hypothetical = p.hypothetical_inputs()
    flags = p.flag_counts()
    target_arm = next((a.id for a in cfg.arms if a.delivery == "lowbarrier"
                       and a.injectable_uptake), cfg.arms[-1].id)

    md: list[str] = []
    A = md.append
    A(f"# Injectable PrEP for people experiencing homelessness: {cfg.run_name}")
    A("")
    A("Generated by `prep_model.report`. Every figure in this document is read from a "
      "run output; none is typed in. Structure follows the CHEERS 2022 checklist.")
    A("")
    A("## Standing caveat")
    A("")
    A(f"**{len(hypothetical)} of the {len(p.accessed)} inputs this run actually read are "
      "flagged `hypothetical`.** They are placeholders with defensible ranges, not local "
      "measurements. Nothing below is a finding about a real city until those rows carry "
      "a source and a data-use agreement. The calibration targets are themselves "
      "hypothetical surveillance summaries.")
    A("")
    A("Three claims are reported separately and must not be substituted for one another: "
      "**health benefit** (infections and QALYs), **cost-effectiveness** (cost per QALY "
      "against a stated threshold), and **cost saving** (whether total spending falls). "
      "An option can pass the first two and fail the third.")
    A("")
    A("Mortality figures are model-estimated deaths. A prevented infection is not a life "
      "saved: modern treatment changes survival and competing causes of death remain.")
    A("")

    A("## 1. Study design, comparators, perspective and horizon")
    A("")
    A(table(["item", "value"], [
        ["population", f"{cfg.n_individuals:,} simulated adults, reported per "
                       f"{cfg.report_per:,} eligible adults where stated"],
        ["comparators", "; ".join(f"{a.id} {a.name}" for a in cfg.arms)],
        ["reference arm", cfg.reference_arm],
        ["primary perspective", persp + " = " + ", ".join(cfg.perspectives[persp])],
        ["other perspectives", ", ".join(k for k in cfg.perspectives if k != persp)],
        ["time step", f"{cfg.step_weeks:g} week(s)"],
        ["simulated horizon", f"{cfg.horizon_years:g} years"],
        ["budget horizon", f"{cfg.budget_years} years"],
        ["beyond the horizon", cfg.raw['time']['post_horizon_policy']],
        ["terminal value", "included" if cfg.terminal_value else "not included"],
        ["discount rate", f"{cfg.discount_rate:.0%} on costs and QALYs; "
                          f"{cfg.raw.get('discount_rates_sensitivity')} tested"],
        ["price year", str(cfg.price_year)],
        ["willingness to pay", ", ".join(money(w) for w in cfg.wtp)],
    ]))
    A("")
    A("Arm E is a matched-contact analytic control: the same outreach, testing and ART "
      "linkage as arm C with no additional injectable uptake. It exists to separate the "
      "effect of the drug from the effect of the contact, and is a modelling "
      "counterfactual, not a proposal to withhold care.")
    A("")

    A("## 2. Table 1. Inputs, sources, uncertainty and assumptions")
    A("")
    A(table_inputs(registry, p))
    A("")
    A(f"Assumption flags across the {len(p.accessed)} inputs read: " +
      ", ".join(f"{k} {v}" for k, v in sorted(flags.items())) + ".")
    if hypothetical:
        A("")
        A("Inputs flagged hypothetical, detected automatically: " +
          ", ".join(f"`{h}`" for h in hypothetical) + ".")
    unused = p.unused()
    if unused:
        A("")
        A("Registry rows this run never read: " + ", ".join(f"`{u}`" for u in unused) + ".")
    A("")

    if calibration:
        A("## 3. Calibration of usual care")
        A("")
        A(f"Rejection ABC over the deterministic model. {calibration['accepted']:,} of "
          f"{calibration['proposals']:,} proposals fell inside every tolerance "
          f"({calibration['acceptance_rate']:.2%}); the ensemble, not any single member, "
          "is the fitted object.")
        A("")
        A(table(["target", "value", "tolerance", "flag"],
                [[t["name"], num(t["value"], 4), num(t["tolerance"], 4), t["flag"]]
                 for t in calibration["targets"]]))
        A("")
        A("Held-out validation targets, scored after the fit:")
        A("")
        A(table(["target", "target value", "ensemble median", "ensemble range",
                 "share within tolerance"],
                [[k, num(v["target"], 4), num(v["ensemble_median"], 4),
                  f"{num(v['ensemble_min'], 4)} to {num(v['ensemble_max'], 4)}",
                  f"{v['within_tolerance_share']:.0%}"]
                 for k, v in calibration["validation"].items()]))
        A("")
        if calibration["unidentified"]:
            A("**Not identified by the targets** and carried through as assumptions rather "
              "than estimates: " +
              ", ".join(f"`{u}`" for u in calibration["unidentified"]) + ".")
            A("")
        for note in calibration["notes"]:
            A("> " + note)
            A("")

    A("## 4. Table 2. What the programme delivered")
    A("")
    A(table_delivery(out, cfg))
    A("")
    A("Contacts, offers and tests are costed whether or not they produce an injection. "
      "An outreach model that only charges for successful doses understates delivery cost.")
    A("")

    A("## 5. Table 3. Health outcomes")
    A("")
    A(table_health(out, cfg, psa))
    A("")
    A(f"Counts and averted counts here are from a single stochastic replicate. "
      f"Table 6 and the conclusion average over all "
      f"{int(cfg.section('uncertainty').get('stochastic_replicates', 1))}, "
      "so the two differ by the replicate noise, "
      "which in this model is large. Where they disagree the averaged figure is "
      "the one to quote, and neither is meaningful without the interval.")
    A("")

    A(f"## 6. Table 4. Annual cash spending, years 1-{cfg.budget_years}")
    A("")
    A("Undiscounted, by cost category and payer. This is a budget-impact statement and is "
      "a different quantity from the discounted lifetime totals in the next table.")
    A("")
    A(table_annual_spending(out, cfg))
    A("")

    A("## 7. Table 5. Discounted costs by arm")
    A("")
    A(f"Perspective: {persp}. Includes the terminal continuation beyond the horizon, which "
      "covers only time strictly after it and uses the same state-specific annual costs the "
      "simulation used; no separate lifetime HIV cost is added on top.")
    A("")
    A(table_lifetime_costs(out, cfg))
    A("")
    A("Ordinary shelter expenditure appears in every arm and largely cancels in the "
      "difference. The HIV-attributable housing increment is set to "
      f"{money(p['cost_hiv_attributable_housing_annual'])} per person-year; it stays at zero "
      "unless a source is supplied, because a prevented infection does not mean a person "
      "stops needing a bed.")
    A("")

    A("## 8. Table 6. Incremental cost-effectiveness")
    A("")
    A(table_cost_effectiveness(out, cfg, psa))
    A("")
    A("Frontier status removes both simple dominance (costlier and no better than a cheaper "
      "arm) and extended dominance (beaten by a mixture of two others).")
    A("")
    A("Results under every perspective:")
    A("")
    rows = []
    for name in cfg.perspectives:
        for c in out.comparisons[name]:
            rows.append([name, c.arm_id, money(c.delta_cost), num(c.delta_qalys, 1),
                         icer_str(c.icer), "yes" if c.cost_saving else "no"])
    A(table(["perspective", "arm", "incremental cost", "incremental QALYs", "ICER",
             "cost saving"], rows))
    A("")

    A("## 9. Table 7. Break-even prices and the retention assumed")
    A("")
    A(table_price_thresholds(out, cfg, grid_rows))
    A("")
    A("Total cost is affine in the acquisition price and nothing in the model reacts to it, "
      "so these are solved exactly from the discounted injection count rather than searched "
      "for. A negative break-even price means the programme does not become cost saving at "
      "any price a manufacturer could charge, and the cost-effectiveness question has to be "
      "answered on its own terms.")
    A("")

    if psa:
        A("## 10. Probabilistic analysis")
        A("")
        first = next(iter(psa.values()))
        A(f"{first.n_draws:,} parameter sets. Parameters the calibration identified are "
          "resampled from the accepted ensemble; everything else is drawn from its registry "
          "distribution. This is decision uncertainty, and it is reported separately from "
          "the Monte Carlo noise of the simulation itself.")
        A("")
        rows = []
        for arm_id, r in psa.items():
            s = r.summary()
            rows.append([arm_id, r.arm_name,
                         money(s["delta_cost"]["mean"]),
                         f"{money(s['delta_cost']['p2_5'])} to {money(s['delta_cost']['p97_5'])}",
                         num(s["delta_qalys"]["mean"], 1),
                         f"{s['probability_cost_saving']:.0%}",
                         f"{s['probability_qaly_gain']:.0%}"] +
                        [f"{s['probability_cost_effective'][str(w)]:.0%}" for w in cfg.wtp])
        A(table(["arm", "name", "mean incremental cost", "95% interval",
                 "mean incremental QALYs", "P(cost saving)", "P(QALY gain)"] +
                [f"P(cost-effective at {money(w)})" for w in cfg.wtp], rows))
        A("")

    if one_way_rows:
        A("## 11. One-way sensitivity")
        A("")
        A(f"Each input at its 2.5th and 97.5th percentile with the rest at base, ranked by "
          f"the swing in incremental net monetary benefit at {money(wtp)} per QALY for arm "
          f"{target_arm}. Ranking by ICER swing is unreadable once an arm crosses into "
          "dominance, which is why net benefit is used.")
        A("")
        A(table(["parameter", "low value", "high value", "NMB at low", "NMB at high", "swing"],
                [[r["parameter"], num(r["low_value"], 4), num(r["high_value"], 4),
                  money(r["low_nmb"]), money(r["high_nmb"]), money(r["swing"])]
                 for r in one_way_rows if "swing" in r]))
        skipped = [r for r in one_way_rows if "error" in r]
        if skipped:
            A("")
            A("Not varied: " + "; ".join(f"`{r['parameter']}` ({r['error']})" for r in skipped) + ".")
        A("")

    if grid_rows:
        A("## 12. Scenario grid")
        A("")
        A("One axis moved at a time with the others at base. A full cross-product of the "
          "five axes is thousands of microsimulations and would add little the one-way "
          "results do not; interactions between axes are therefore not shown.")
        A("")
        A(table(["axis", "value", "incremental cost", "incremental QALYs",
                 "infections averted", "ICER", "cost saving"],
                [[g["axis"], f"{g['value']:g}", money(g["delta_cost"]),
                  num(g["delta_qalys"], 1), num(g["infections_averted"], 0),
                  icer_str(g["icer"]), "yes" if g["cost_saving"] else "no"]
                 for g in grid_rows]))
        A("")

    if structural:
        A("## 13. Structural sensitivity")
        A("")
        A("These scenarios change what the model assumes, not what it is given. "
          "`direct_effects_only` freezes infectious prevalence at baseline, removing herd "
          "effects; it is the conservative bound on benefit. `injection_route_efficacy` "
          "relaxes the base case's deliberately conservative assumption that injectable "
          "PrEP does nothing against equipment-sharing acquisition, which is a structural "
          "choice and not a finding of zero biological effect.")
        A("")
        rows = []
        for name, so in structural.items():
            for c in so.comparisons[persp]:
                if c.arm_id != target_arm:
                    continue
                rows.append([name, money(c.delta_cost), num(c.delta_qalys, 1),
                             num(c.infections_averted, 0), icer_str(c.icer),
                             "yes" if c.cost_saving else "no"])
        base_c = next(c for c in out.comparisons[persp] if c.arm_id == target_arm)
        rows.insert(0, ["base case", money(base_c.delta_cost), num(base_c.delta_qalys, 1),
                        num(base_c.infections_averted, 0), icer_str(base_c.icer),
                        "yes" if base_c.cost_saving else "no"])
        A(f"Arm {target_arm} against arm {cfg.reference_arm}:")
        A("")
        A(table(["scenario", "incremental cost", "incremental QALYs", "infections averted",
                 "ICER", "cost saving"], rows))
        A("")

    A("## 14. Population accounting")
    A("")
    A(table(["arm", "allocated", "entered", "deaths", "out-migration",
             "alive and resident at end", "never entered"],
            [[a.id] + [num(out.arms[a.id].accounting[k], 0) for k in
                       ("allocated", "entered", "deaths", "migrations",
                        "alive_and_resident_at_end", "not_yet_entered")]
             for a in cfg.arms]))
    A("")

    A("## 15. Conclusion")
    A("")
    A(conclusion(out, cfg, target_arm))
    A("")
    A("That sentence is a statement about the model under its stated inputs. It is not "
      "evidence about any real programme, and it must not be quoted without the "
      "hypothetical-input list above.")
    A("")

    A("## 16. Limitations")
    A("")
    for lim in [
        "Calibration targets and several delivery inputs are hypothetical; the model is "
        "internally validated but not externally validated against local data.",
        "Free parameters outnumber calibration targets, so the fit is under-determined by "
        "construction and an ensemble is retained rather than a point estimate.",
        "Mixing between exposure groups is a two-parameter approximation; assortativity is "
        "assumed, not measured.",
        "Injectable efficacy against equipment-sharing acquisition is set to zero as a "
        "conservative structural assumption and relaxed only in sensitivity analysis.",
        "Housing transitions are a three-state Markov process calibrated to a stationary "
        "distribution, not to observed individual trajectories.",
        "The terminal value is an exponential-survival annuity, which assumes a constant "
        "hazard after the horizon.",
        "Out-migration removes people from the model; their later outcomes are unobserved "
        "and uncosted. This is not a small correction and it is not symmetric: at "
        f"{p['migration_rate_per_year']:.0%} a year the treatment cost of an infection is "
        f"worth {_years(p, cfg, with_migration=True):.1f} "
        f"discounted years against {_years(p, cfg, with_migration=False):.1f} "
        "on mortality alone, so the model recovers a fraction of the lifetime cost an "
        "averted infection would really avoid, and it loses it from the arm that averts "
        "infections. The bias runs against prevention.",
        "The acquisition price used is an announced list price, not a negotiated contract "
        "price, and it is the single input that moves the cost conclusion most.",
    ]:
        A(f"- {lim}")
    A("")

    A("## 17. Provenance")
    A("")
    A(table(["item", "value"], [[k, str(v)] for k, v in prov.to_json().items()]))
    A("")

    path = out_dir / "report.md"
    path.write_text("\n".join(md))
    (out_dir / "provenance.json").write_text(json.dumps(prov.to_json(), indent=2))
    return path
