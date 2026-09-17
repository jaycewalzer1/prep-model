"""``python -m prep_model.validate --config config/base.yaml``

Checks the inputs before anything is run with them: the registry's structure,
the config's internal consistency, and whether the model's own health depends on
numbers nobody has measured. Exits non-zero only on problems that make a run
meaningless, and prints the hypothetical inputs as a warning either way.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .deterministic import FREE
from .params import Registry
from .simulate import run_all_arms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prep_model.validate")
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="also run one short simulation of every arm")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    registry = Registry.load()
    problems = registry.structural_audit()

    calib = cfg.section("calibration")
    for name in calib.get("free_parameters") or FREE:
        if name not in registry.rows:
            problems.append(f"calibration frees {name!r}, which is not in the registry")
        elif registry.rows[name].distribution == "fixed":
            problems.append(f"calibration frees {name!r}, which has a fixed distribution "
                            "and so has no range to search")
    n_free = len(calib.get("free_parameters") or FREE)
    n_targets = len(calib.get("targets") or {})

    p = registry.resolve(cfg.price_year, overrides=cfg.overrides)
    print(f"config              {cfg.path}")
    print(f"registry rows       {len(registry.rows)}")
    print(f"arms                {', '.join(a.id + ' ' + a.name for a in cfg.arms)}")
    print(f"price year          {cfg.price_year}")
    print(f"horizon             {cfg.horizon_years:g} years at {cfg.step_weeks:g}-week steps "
          f"({cfg.n_steps} steps)")
    print(f"population          {cfg.n_individuals:,}")
    print(f"calibration         {n_free} free parameters against {n_targets} targets")
    if n_free > n_targets:
        print("                    under-determined by construction; an ensemble is the result")

    if args.smoke:
        short = cfg.copy_with(population={"n_individuals": 500},
                              time={"horizon_years": 1.0})
        results = run_all_arms(short, p)
        print("\nsmoke run (500 people, 1 year):")
        for arm_id, r in results.items():
            print(f"  {arm_id} {r.arm_name:26s} infections {r.epi['infections']:6.0f}  "
                  f"doses {r.counters.doses:6d}  QALYs {r.qalys:9.1f}")
        for arm_id, r in results.items():
            a = r.accounting
            balance = a["entered"] - a["deaths"] - a["migrations"] - a["alive_and_resident_at_end"]
            if abs(balance) > 0.5:
                problems.append(f"arm {arm_id}: population accounting is off by {balance:g}")

    hypo = p.hypothetical_inputs() if p.accessed else [
        n for n, r in registry.rows.items() if r.assumption_flag == "hypothetical"]
    print(f"\nhypothetical inputs {len(hypo)}")
    for h in hypo:
        print(f"  {h}: {registry.rows[h].transformation or registry.rows[h].definition}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for x in problems:
            print(f"  - {x}")
        return 1
    print("\nno structural problems found. This checks internal consistency only; "
          "it says nothing about whether the inputs are true of any real place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
