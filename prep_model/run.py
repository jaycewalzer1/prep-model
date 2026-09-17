"""``python -m prep_model.run --config config/base.yaml --all-arms``

Runs the policy experiments and writes machine-readable outputs. Analysis and
report generation are a separate command, so a long run is never repeated to
change a table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import calibration as calib_mod
from . import experiments
from .config import ROOT, load_config
from .params import Registry
from .report import provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prep_model.run")
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--all-arms", action="store_true",
                    help="kept for interface compatibility; every arm always runs")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--calibration", default=None,
                    help="path to calibration.json; its ensemble is used for the PSA")
    ap.add_argument("--psa-draws", type=int, default=None)
    ap.add_argument("--skip", default="", help="comma-separated: psa,one_way,two_way,grid,structural")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    registry = Registry.load()
    p = registry.resolve(cfg.price_year, overrides=cfg.overrides)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    unc = cfg.section("uncertainty")
    reps = int(unc.get("stochastic_replicates", 1))
    n_draws = args.psa_draws if args.psa_draws is not None else int(unc.get("parameter_draws", 0))
    exp = cfg.section("experiments")
    target = next((a.id for a in cfg.arms if a.delivery == "lowbarrier" and a.injectable_uptake),
                  cfg.arms[-1].id)

    ensemble = None
    calib_path = Path(args.calibration) if args.calibration else out_dir / "calibration.json"
    if calib_path.exists():
        ensemble = calib_mod.load(calib_path).ensemble
        print(f"using a calibrated ensemble of {len(ensemble)} members from {calib_path}")
    else:
        print(f"no calibration at {calib_path}: running at registry base values and "
              "sampling every varying parameter from its prior")

    def stage(name: str, fn):
        if name in skip:
            print(f"  {name:12s} skipped")
            return None
        t = time.time()
        value = fn()
        print(f"  {name:12s} {time.time() - t:6.1f}s")
        return value

    print("running:")
    base = stage("base", lambda: experiments.run_scenario(cfg, p, "base", replicates=reps))
    (out_dir / "base.json").write_text(json.dumps(base.to_json(), indent=2))

    psa = stage("psa", lambda: experiments.run_psa(cfg, registry, n_draws, ensemble)) \
        if n_draws else None
    if psa:
        (out_dir / "psa.json").write_text(json.dumps(
            {k: v.summary() for k, v in psa.items()}, indent=2))
        (out_dir / "psa_draws.json").write_text(json.dumps(
            {k: {"delta_cost": v.delta_cost, "delta_qalys": v.delta_qalys,
                 "infections_averted": v.infections_averted} for k, v in psa.items()}))

    ow = stage("one_way", lambda: experiments.one_way(
        cfg, registry, list(exp.get("one_way") or []), target, replicates=1))
    if ow:
        (out_dir / "one_way.json").write_text(json.dumps(ow, indent=2))

    pair = (exp.get("two_way") or [[None, None]])[0]
    tw = None
    if pair[0]:
        tw = stage("two_way", lambda: experiments.two_way(
            cfg, registry, pair[0], pair[1], target, n=4, replicates=1))
        if tw:
            (out_dir / "two_way.json").write_text(json.dumps(
                {"names": pair, "grid": tw}, indent=2))

    grid = stage("grid", lambda: experiments.scenario_grid(cfg, registry, target, replicates=1))
    if grid:
        (out_dir / "scenario_grid.json").write_text(json.dumps(grid, indent=2))

    struct = stage("structural", lambda: experiments.structural_scenarios(
        cfg, registry, replicates=1))
    if struct:
        (out_dir / "structural.json").write_text(json.dumps(
            {k: v.to_json() for k, v in struct.items()}, indent=2))

    manifest = {
        "run_name": cfg.run_name,
        "config": str(cfg.path),
        "provenance": provenance(cfg).to_json(),
        "replicates": reps,
        "psa_draws": n_draws if psa else 0,
        "calibration_used": str(calib_path) if ensemble else None,
        "target_arm": target,
        "two_way_names": pair if tw else None,
        "accessed_parameters": sorted(p.accessed),
        "hypothetical_inputs": p.hypothetical_inputs(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwritten to {out_dir}")
    print(f"next: python -m prep_model.analyze --run-dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
