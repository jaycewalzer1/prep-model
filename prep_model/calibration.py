"""Fitting usual care before looking at any intervention outcome.

Rejection ABC over the deterministic model. Proposals are drawn from the
registry's own priors, so a calibrated value can never wander outside the range
the evidence table is willing to defend.

Three things this is built to avoid.

Diagnoses are not incidence. The deterministic model carries an explicit
testing process, so a diagnosis count is produced by infection, testing and
delay rather than by dividing a prevalence by a number of years.

A single best fit is not retained. An ensemble is, because a single point
estimate manufactures precision the data does not contain.

A parameter the targets cannot determine is flagged rather than reported. If
the accepted range is nearly as wide as the prior, the targets did not identify
it and the report says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import deterministic
from .config import Config
from .params import Params, Registry
from .rng import Stream, uniform


@dataclass
class Target:
    name: str
    value: float
    tolerance: float
    flag: str = "hypothetical"


@dataclass
class CalibrationResult:
    free_parameters: list[str]
    targets: list[Target]
    validation_targets: list[Target]
    ensemble: list[dict[str, float]]
    proposals: int
    accepted: int
    acceptance_rate: float
    prior_ranges: dict[str, tuple[float, float]]
    posterior_ranges: dict[str, tuple[float, float]]
    unidentified: list[str]
    validation: dict[str, dict[str, float]]
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "free_parameters": self.free_parameters,
            "targets": [t.__dict__ for t in self.targets],
            "validation_targets": [t.__dict__ for t in self.validation_targets],
            "proposals": self.proposals,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "prior_ranges": {k: list(v) for k, v in self.prior_ranges.items()},
            "posterior_ranges": {k: list(v) for k, v in self.posterior_ranges.items()},
            "unidentified": self.unidentified,
            "validation": self.validation,
            "notes": self.notes,
            "ensemble": self.ensemble,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))

    def median_member(self) -> dict[str, float]:
        if not self.ensemble:
            return {}
        return {k: float(torch.tensor([m[k] for m in self.ensemble]).median())
                for k in self.free_parameters}


def _targets(cfg: Config, key: str) -> list[Target]:
    section = cfg.section("calibration").get(key) or {}
    return [Target(name=n, value=float(v["value"]), tolerance=float(v["tolerance"]),
                   flag=v.get("flag", "hypothetical"))
            for n, v in section.items()]


def calibrate(cfg: Config, registry: Registry, p: Params,
              years: float = 10.0) -> CalibrationResult:
    section = cfg.section("calibration")
    free = list(section.get("free_parameters") or deterministic.FREE)
    targets = _targets(cfg, "targets")
    validation = _targets(cfg, "validation_targets")
    n_prop = int(section.get("proposals", 2000))
    keep = int(section.get("accepted_ensemble_size", 50))

    notes: list[str] = []
    if len(free) > len(targets):
        notes.append(
            f"{len(free)} free parameters against {len(targets)} calibration targets: "
            "the fit is under-determined by construction and the ensemble, not any "
            "single member, is the result."
        )

    # Proposals are uniforms on each prior's CDF, drawn from the shared stream
    # machinery so a calibration run reproduces exactly.
    idx = torch.arange(n_prop, dtype=torch.int64)
    draws = {name: uniform(cfg.seed, Stream.ATTR_HIV, 77_000 + i, idx)
             for i, name in enumerate(free)}
    batch = {name: torch.tensor([registry.rows[name].quantile(float(u)) for u in draws[name]],
                                dtype=torch.float64)
             for name in free}

    out = deterministic.run(p, years, cfg.dt, n_people=1000.0, batch=batch).as_dict()

    distance = torch.zeros(n_prop, dtype=torch.float64)
    for t in targets:
        if t.name not in out:
            raise KeyError(f"calibration target {t.name!r} is not produced by the model")
        distance = torch.maximum(distance, (out[t.name] - t.value).abs() / t.tolerance)

    accepted_mask = distance <= 1.0
    order = torch.argsort(distance)
    accepted_idx = order[accepted_mask[order]][:keep]
    n_accepted = int(accepted_mask.sum())
    if n_accepted == 0:
        notes.append(
            "No proposal fell inside every tolerance. The ensemble below is the closest "
            "set by maximum normalised distance and must not be described as a fit."
        )
        accepted_idx = order[:keep]

    ensemble = [{name: float(batch[name][i]) for name in free} for i in accepted_idx.tolist()]

    prior_ranges = {name: (float(registry.rows[name].quantile(0.025)),
                           float(registry.rows[name].quantile(0.975))) for name in free}
    posterior_ranges = {}
    unidentified = []
    for name in free:
        vals = torch.tensor([m[name] for m in ensemble], dtype=torch.float64)
        lo, hi = float(vals.min()), float(vals.max())
        posterior_ranges[name] = (lo, hi)
        prior_width = prior_ranges[name][1] - prior_ranges[name][0]
        if prior_width > 0 and (hi - lo) / prior_width > 0.8:
            unidentified.append(name)
    if unidentified:
        notes.append(
            "These inputs are not determined by the available targets and should be read as "
            "assumptions carried through the analysis, not as estimates: "
            + ", ".join(unidentified) + "."
        )

    validation_report: dict[str, dict[str, float]] = {}
    for t in validation:
        if t.name not in out:
            continue
        vals = out[t.name][accepted_idx]
        validation_report[t.name] = {
            "target": t.value,
            "tolerance": t.tolerance,
            "ensemble_median": float(vals.median()),
            "ensemble_min": float(vals.min()),
            "ensemble_max": float(vals.max()),
            "within_tolerance_share": float(((vals - t.value).abs() <= t.tolerance)
                                            .to(torch.float64).mean()),
        }

    return CalibrationResult(
        free_parameters=free,
        targets=targets,
        validation_targets=validation,
        ensemble=ensemble,
        proposals=n_prop,
        accepted=n_accepted,
        acceptance_rate=n_accepted / n_prop,
        prior_ranges=prior_ranges,
        posterior_ranges=posterior_ranges,
        unidentified=unidentified,
        validation=validation_report,
        notes=notes,
    )


def load(path: Path) -> CalibrationResult:
    raw = json.loads(Path(path).read_text())
    return CalibrationResult(
        free_parameters=raw["free_parameters"],
        targets=[Target(**t) for t in raw["targets"]],
        validation_targets=[Target(**t) for t in raw["validation_targets"]],
        ensemble=raw["ensemble"],
        proposals=raw["proposals"],
        accepted=raw["accepted"],
        acceptance_rate=raw["acceptance_rate"],
        prior_ranges={k: tuple(v) for k, v in raw["prior_ranges"].items()},
        posterior_ranges={k: tuple(v) for k, v in raw["posterior_ranges"].items()},
        unidentified=raw["unidentified"],
        validation=raw["validation"],
        notes=raw["notes"],
    )
