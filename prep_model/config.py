"""Scenario configuration.

Structure lives here (arms, horizons, perspectives, grids). Numbers live in the
evidence registry. The one exception is ``parameter_overrides``, which is how a
scenario pins a registry value, and which the report prints verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Arm:
    id: str
    name: str
    delivery: str          # none | clinic | lowbarrier
    product: str           # none | lenacapavir | cabotegravir
    injectable_uptake: bool
    description: str = ""

    @property
    def has_outreach(self) -> bool:
        return self.delivery != "none"


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    # -- convenience accessors -------------------------------------------------
    @property
    def run_name(self) -> str:
        return self.raw["run_name"]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def device(self) -> str:
        return self.raw.get("device", "cpu")

    @property
    def price_year(self) -> int:
        return int(self.raw["price_year"])

    @property
    def discount_rate(self) -> float:
        return float(self.raw["discount_rate"])

    @property
    def wtp(self) -> list[float]:
        return [float(x) for x in self.raw["willingness_to_pay"]]

    @property
    def n_individuals(self) -> int:
        return int(self.raw["population"]["n_individuals"])

    @property
    def report_per(self) -> int:
        return int(self.raw["population"].get("report_per", 1000))

    @property
    def step_weeks(self) -> float:
        return float(self.raw["time"]["step_weeks"])

    @property
    def dt(self) -> float:
        """Step length in years."""
        return self.step_weeks / 52.0

    @property
    def horizon_years(self) -> float:
        return float(self.raw["time"]["horizon_years"])

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon_years * 52.0 / self.step_weeks))

    @property
    def budget_years(self) -> int:
        return int(self.raw["time"]["budget_years"])

    @property
    def terminal_value(self) -> bool:
        return bool(self.raw["time"].get("terminal_value", True))

    @property
    def arms(self) -> list[Arm]:
        return [Arm(**a) for a in self.raw["arms"]]

    @property
    def reference_arm(self) -> str:
        return self.raw["reference_arm"]

    @property
    def payers(self) -> dict[str, str]:
        return dict(self.raw["payers"])

    @property
    def perspectives(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self.raw["perspectives"].items()}

    @property
    def primary_perspective(self) -> str:
        return self.raw["primary_perspective"]

    @property
    def overrides(self) -> dict[str, float]:
        return {k: float(v) for k, v in (self.raw.get("parameter_overrides") or {}).items()}

    def arm(self, arm_id: str) -> Arm:
        for a in self.arms:
            if a.id == arm_id:
                return a
        raise KeyError(f"no arm with id {arm_id!r}")

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name) or {})

    def copy_with(self, **patch: Any) -> "Config":
        raw = _deep_merge(self.raw, patch)
        return Config(raw, self.path)


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    raw = yaml.safe_load(path.read_text())
    parent = raw.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        raw = _deep_merge(base.raw, raw)
    _validate(raw, path)
    return Config(raw, path)


def _validate(raw: dict, path: Path) -> None:
    required = ["run_name", "seed", "price_year", "discount_rate", "willingness_to_pay",
                "population", "time", "arms", "reference_arm", "payers", "perspectives",
                "primary_perspective"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"{path}: config is missing {missing}")
    arm_ids = [a["id"] for a in raw["arms"]]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError(f"{path}: duplicate arm ids {arm_ids}")
    if raw["reference_arm"] not in arm_ids:
        raise ValueError(f"{path}: reference_arm {raw['reference_arm']} is not an arm")
    if raw["primary_perspective"] not in raw["perspectives"]:
        raise ValueError(f"{path}: primary_perspective is not a defined perspective")
    categories = set(raw["payers"])
    for name, cats in raw["perspectives"].items():
        unknown = set(cats) - categories
        if unknown:
            raise ValueError(f"{path}: perspective {name} names unknown cost categories {sorted(unknown)}")
    for a in raw["arms"]:
        if a["delivery"] not in {"none", "clinic", "lowbarrier"}:
            raise ValueError(f"{path}: arm {a['id']} has unknown delivery {a['delivery']!r}")
        if a["product"] not in {"none", "lenacapavir", "cabotegravir"}:
            raise ValueError(f"{path}: arm {a['id']} has unknown product {a['product']!r}")
        if a["injectable_uptake"] and a["product"] == "none":
            raise ValueError(f"{path}: arm {a['id']} takes up an injectable but names no product")
