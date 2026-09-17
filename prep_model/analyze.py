"""``python -m prep_model.analyze --run-dir outputs/main``

Turns a run directory into tables and figures. The expensive sweeps are read
back from disk, but the base case is re-simulated from the recorded seed and
config and then checked against what the run wrote. If the two disagree the
report is not written: a report that silently describes a different run than
the one on disk is worse than no report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import experiments
from .config import load_config
from .experiments import PSAResult
from .params import Registry
from .report import build_report

REPRODUCIBILITY_TOLERANCE = 1e-6


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prep_model.analyze")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    manifest = _load(run_dir / "manifest.json")
    if manifest is None:
        print(f"no manifest.json in {run_dir}: run prep_model.run first")
        return 1

    cfg = load_config(manifest["config"])
    registry = Registry.load()
    p = registry.resolve(cfg.price_year, overrides=cfg.overrides)

    print("re-simulating the base case from the recorded seed and config")
    base = experiments.run_scenario(cfg, p, "base", replicates=manifest["replicates"])

    recorded = _load(run_dir / "base.json")
    persp = cfg.primary_perspective
    for c in base.comparisons[persp]:
        was = next(d for d in recorded["comparisons"][persp] if d["arm_id"] == c.arm_id)
        for field in ("delta_cost", "delta_qalys", "infections_averted"):
            a, b = getattr(c, field), was[field]
            if abs(a - b) > REPRODUCIBILITY_TOLERANCE * max(1.0, abs(b)):
                print(f"REPRODUCIBILITY FAILURE: arm {c.arm_id} {field} is {a} now and was "
                      f"{b} when the run was written. The run directory and the current "
                      f"code or data do not agree; no report written.")
                return 2
    print("  matches base.json")

    psa_raw = _load(run_dir / "psa_draws.json")
    psa = None
    if psa_raw:
        psa = {}
        for arm_id, d in psa_raw.items():
            arm_name = base.arms[arm_id].arm_name
            r = PSAResult(arm_id=arm_id, arm_name=arm_name, perspective=persp,
                          n_draws=len(d["delta_cost"]), delta_cost=d["delta_cost"],
                          delta_qalys=d["delta_qalys"],
                          infections_averted=d["infections_averted"],
                          nmb={w: [w * q - c for c, q in
                                   zip(d["delta_cost"], d["delta_qalys"])] for w in cfg.wtp})
            psa[arm_id] = r

    two_way = _load(run_dir / "two_way.json")
    structural_raw = _load(run_dir / "structural.json")
    structural = None
    if structural_raw:
        # Structural scenarios are re-simulated too: the report needs the ledgers,
        # and each is one paired run rather than a sweep.
        print("re-simulating structural scenarios")
        structural = experiments.structural_scenarios(cfg, registry, replicates=1)

    calibration = _load(run_dir / "calibration.json")

    path = build_report(
        cfg, registry, p, base, run_dir,
        calibration=calibration,
        psa=psa,
        one_way_rows=_load(run_dir / "one_way.json"),
        two_way_grid=two_way["grid"] if two_way else None,
        two_way_names=tuple(two_way["names"]) if two_way else None,
        grid_rows=_load(run_dir / "scenario_grid.json"),
        structural=structural,
    )
    print(f"\nwritten {path}")
    print(f"figures in {run_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
