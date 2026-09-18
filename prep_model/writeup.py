"""Assemble the figures and tables the README quotes, from saved runs.

This module runs no simulation. It reads the JSON a completed run already wrote
and produces the tracked ``figures/`` directory, so the write-up can be rebuilt
and checked without a several-hour recompute. If a number in the README and a
number here ever disagree, this is the one that is current.

Two of the figures are new here rather than in ``report``: the break-even price
across accounting bases, and the elimination ladder. Both are comparisons
*between* runs or between framings, which is not something the per-run report is
in a position to draw.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .analytic import InfectionBurden, break_even_price_bases, lifetime_burden_of_one_infection
from .config import Config, load_config
from .params import Registry

REPORT_FIGURES = (
    "fig1_cumulative_infections.png",
    "fig2_annual_spending.png",
    "fig3_ce_plane_ceac.png",
    "fig4_net_cost_heatmap.png",
    "fig5_tornado.png",
)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def price_bases(run_dir: Path, cfg: Config, burden: InfectionBurden,
                wtp: float = 100_000.0) -> list:
    """Break-even dose prices for every arm in a saved run that buys doses."""
    d = json.loads((run_dir / "base.json").read_text())
    persp = cfg.primary_perspective
    # Arm labels come from the configuration, never from a table here. A
    # hand-kept copy of them drifts, and an arm captioned as the wrong product is
    # not a cosmetic error.
    names = {a.id: a.name.replace("_", " ") for a in cfg.arms}
    out = []
    for comp in d["comparisons"][persp]:
        arm = comp["arm_id"]
        pr = d["prices"].get(arm)
        if not pr or pr.get("cost_neutral_price") is None:
            continue
        programme = sum(v for k, v in comp["delta_cost_by_category"].items()
                        if k.startswith("program_"))
        out.append(break_even_price_bases(
            arm_id=arm,
            label=names.get(arm, arm),
            current_price=pr["current_price"],
            discounted_doses=pr["discounted_doses"],
            infections_averted=comp["infections_averted"],
            programme_cost=programme,
            budget_price_as_modelled=pr["cost_neutral_price"],
            burden=burden,
            wtp=wtp,
        ))
    return out


def figure_break_even(bases_short: list, bases_long: list, path: Path) -> Path:
    """What an averted infection is allowed to be worth decides the answer.

    Each arm is drawn as a span from the price the model's own accounting
    supports up to the price a payer who bears the whole lifetime cost could
    justify, against a marker for what the dose costs today. An arm whose span
    reaches past its marker pays for itself.
    """
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, bases, title in ((axes[0], bases_short, "5-year horizon"),
                             (axes[1], bases_long, "40-year horizon")):
        labels = [f"{b.arm_id} {b.label}" for b in bases]
        ys = range(len(bases))
        for y, b in zip(ys, bases):
            lo = min(b.budget_price_as_modelled, b.budget_price_lifetime)
            hi = max(b.threshold_price_lifetime, b.budget_price_lifetime)
            ax.plot([lo, hi], [y, y], lw=8, alpha=0.35, color="tab:blue",
                    solid_capstyle="butt")
            ax.plot([b.budget_price_as_modelled], [y], "o", color="tab:red", ms=8,
                    label="budget, as modelled" if y == 0 else None)
            ax.plot([b.budget_price_lifetime], [y], "s", color="tab:orange", ms=8,
                    label="budget, lifetime cost" if y == 0 else None)
            ax.plot([b.threshold_price_lifetime], [y], "D", color="tab:green", ms=8,
                    label="$100k/QALY, lifetime" if y == 0 else None)
            ax.plot([b.current_price], [y], "*", color="black", ms=16,
                    label="price today" if y == 0 else None)
        ax.axvline(0, color="grey", lw=0.8, ls=":")
        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("break-even price per dose (USD)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Break-even dose price depends on what an averted infection "
                 "is allowed to be worth\n(a star to the left of a marker means "
                 "the arm pays for itself at today's price)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_elimination(elim: dict, path: Path) -> Path:
    """Where incidence stops falling, and what is still standing when it does."""
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    rungs = elim["rungs"]
    xs = [r["reach_multiple"] for r in rungs]
    ax1.plot(xs, [r["incidence_per_100py"] for r in rungs], "o-", color="tab:blue")
    ax1.axhline(elim["reference_incidence_per_100py"], color="tab:red", ls="--",
                lw=1, label="usual care")
    ax1.axhline(0.0, color="black", lw=0.8)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"x{x:g}" for x in xs])
    ax1.set_xlabel("outreach reach, capacity never binding")
    ax1.set_ylabel("incidence per 100 person-years")
    ax1.set_title("Saturating delivery does not reach zero")
    ax1.legend(fontsize=8)

    floors = elim["floors"]
    names = [f["label"] for f in floors]
    vals = [f["infections"] for f in floors]
    ax2.barh(range(len(floors)), vals, color="tab:purple", alpha=0.75)
    ax2.set_yticks(range(len(floors)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("infections remaining at the top rung")
    ax2.set_title("Each row removes a barrier, keeping the ones above it", fontsize=10)
    for i, v in enumerate(vals):
        ax2.text(v, i, f" {v:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def render_price_table(bases: list) -> str:
    lines = ["| arm | price today | budget, as modelled | budget, lifetime | "
             "$100k/QALY, lifetime | cut to reach $100k col. |",
             "|---|---|---|---|---|---|"]
    for b in bases:
        lines.append(
            f"| {b.arm_id} {b.label} | ${b.current_price:,.0f} | "
            f"${b.budget_price_as_modelled:,.0f} | ${b.budget_price_lifetime:,.0f} | "
            f"${b.threshold_price_lifetime:,.0f} | {b.discount_required:.0%} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--short", default="outputs/dev", help="the 5-year run directory")
    ap.add_argument("--long", default="outputs/lifetime", help="the 40-year run directory")
    ap.add_argument("--short-config", default="config/dev.yaml")
    ap.add_argument("--long-config", default="config/lifetime.yaml")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--wtp", type=float, default=100_000.0)
    a = ap.parse_args(argv)

    fig_dir = Path(a.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    short_dir, long_dir = Path(a.short), Path(a.long)

    cfg_s = load_config(a.short_config)
    cfg_l = load_config(a.long_config)
    p = Registry.load().resolve(cfg_s.price_year, overrides=cfg_s.overrides)
    burden = lifetime_burden_of_one_infection(p, cfg_s.discount_rate)

    # The per-run report already drew these; copy rather than redraw so there is
    # exactly one piece of code that knows how they are made.
    copied = []
    for name in REPORT_FIGURES:
        src = short_dir / "figures" / name
        if src.exists():
            shutil.copyfile(src, fig_dir / name)
            copied.append(name)
    print(f"copied {len(copied)} report figures from {short_dir / 'figures'}")

    bases_s = price_bases(short_dir, cfg_s, burden, a.wtp)
    bases_l = price_bases(long_dir, cfg_l, burden, a.wtp) if long_dir.exists() else []
    if bases_l:
        figure_break_even(bases_s, bases_l, fig_dir / "fig6_break_even_price.png")
        print("wrote fig6_break_even_price.png")

    elim_path = short_dir / "elimination.json"
    if elim_path.exists():
        figure_elimination(json.loads(elim_path.read_text()),
                           fig_dir / "fig7_elimination.png")
        print("wrote fig7_elimination.png")

    print(f"\nlifetime burden of one infection: net cost ${burden.net_cost:,.0f}, "
          f"{burden.qalys_lost:.2f} QALYs, ${burden.value_at(a.wtp):,.0f} at "
          f"${a.wtp:,.0f}/QALY\n")
    print("5-year horizon\n" + render_price_table(bases_s))
    if bases_l:
        print("\n40-year horizon\n" + render_price_table(bases_l))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
