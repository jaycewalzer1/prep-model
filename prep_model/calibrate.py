"""``python -m prep_model.calibrate --config config/base.yaml``

Fits usual care and writes the ensemble to ``outputs/<run>/calibration.json``.
Nothing about the intervention is touched here, and nothing here is allowed to
look at an intervention outcome.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .calibration import calibrate
from .config import ROOT, load_config
from .params import Registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prep_model.calibrate")
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    registry = Registry.load()
    p = registry.resolve(cfg.price_year, overrides=cfg.overrides)

    result = calibrate(cfg, registry, p)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / cfg.run_name
    path = out_dir / "calibration.json"
    result.save(path)

    print(f"proposals           {result.proposals:,}")
    print(f"accepted            {result.accepted:,} ({result.acceptance_rate:.2%})")
    print(f"ensemble retained   {len(result.ensemble)}")
    print("\nposterior range as a share of the prior range:")
    for name in result.free_parameters:
        lo, hi = result.posterior_ranges[name]
        plo, phi = result.prior_ranges[name]
        share = (hi - lo) / (phi - plo) if phi > plo else float("nan")
        mark = "  <- not identified" if name in result.unidentified else ""
        print(f"  {name:32s} {lo:10.4g} to {hi:10.4g}   {share:5.1%}{mark}")

    print("\nheld-out validation targets:")
    for name, v in result.validation.items():
        print(f"  {name:28s} target {v['target']:.4g} +/- {v['tolerance']:.4g}; "
              f"ensemble median {v['ensemble_median']:.4g}; "
              f"{v['within_tolerance_share']:.0%} of the ensemble inside tolerance")

    for note in result.notes:
        print(f"\nNOTE: {note}")
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
