"""The evidence registry.

Every number the model uses comes from ``data/parameters.csv``. Nothing is a
Python constant, because a constant hides its units, its price year, its source
and whether anybody ever measured it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ASSUMPTION_FLAGS = {"observed_local", "transported", "calibrated", "hypothetical"}
DISTRIBUTIONS = {"fixed", "uniform", "triangular", "lognormal"}
REQUIRED_COLUMNS = [
    "name",
    "definition",
    "value",
    "units",
    "population",
    "geography",
    "study_year",
    "currency_year",
    "source_url",
    "uncertainty_distribution",
    "lower_bound",
    "upper_bound",
    "evidence_grade",
    "transformation",
    "assumption_flag",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class ParameterError(KeyError):
    pass


@dataclass(frozen=True)
class ParameterRow:
    name: str
    definition: str
    value: float
    units: str
    population: str
    geography: str
    study_year: str
    currency_year: str
    source_url: str
    distribution: str
    lower: float | None
    upper: float | None
    evidence_grade: str
    transformation: str
    assumption_flag: str

    @property
    def is_money(self) -> bool:
        return self.units.strip().upper() == "USD"

    def quantile(self, u: float) -> float:
        """Inverse CDF at ``u``. ``u = 0.5`` is not necessarily ``value``."""
        d = self.distribution
        if d == "fixed":
            return self.value
        a, b = self.lower, self.upper
        if a is None or b is None:
            raise ParameterError(f"{self.name}: distribution {d} needs both bounds")
        u = min(max(u, 1e-9), 1 - 1e-9)
        if d == "uniform":
            return a + u * (b - a)
        if d == "triangular":
            c = self.value
            if not (a <= c <= b):
                raise ParameterError(f"{self.name}: mode {c} outside [{a}, {b}]")
            fc = (c - a) / (b - a) if b > a else 0.0
            if u < fc:
                return a + math.sqrt(u * (b - a) * (c - a))
            return b - math.sqrt((1 - u) * (b - a) * (b - c))
        if d == "lognormal":
            if a <= 0:
                raise ParameterError(f"{self.name}: lognormal needs a positive lower bound")
            sigma = math.log(b / a) / (2 * 1.959963984540054)
            return self.value * math.exp(sigma * _norm_ppf(u))
        raise ParameterError(f"{self.name}: unknown distribution {d}")


def _norm_ppf(u: float) -> float:
    """Acklam's rational approximation to the standard normal quantile."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if u < plow:
        q = math.sqrt(-2 * math.log(u))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if u > phigh:
        q = math.sqrt(-2 * math.log(1 - u))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = u - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _opt_float(x) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)) or x == "":
        return None
    return float(x)


def _opt_str(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


class PriceIndex:
    """Converts money between price years."""

    def __init__(self, table: pd.DataFrame):
        self._idx = {int(r.year): float(r.index) for r in table.itertuples()}

    @classmethod
    def load(cls, path: Path | None = None) -> "PriceIndex":
        return cls(pd.read_csv(path or DATA_DIR / "price_index.csv"))

    def factor(self, from_year: int, to_year: int) -> float:
        for y in (from_year, to_year):
            if y not in self._idx:
                raise ParameterError(f"price index has no entry for {y}")
        return self._idx[to_year] / self._idx[from_year]


class Registry:
    """The parameter table as loaded, before any price year or draw is applied."""

    def __init__(self, rows: dict[str, ParameterRow], prices: PriceIndex):
        self.rows = rows
        self.prices = prices

    @classmethod
    def load(cls, path: Path | None = None, price_path: Path | None = None) -> "Registry":
        table = pd.read_csv(path or DATA_DIR / "parameters.csv")
        missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
        if missing:
            raise ParameterError(f"parameters.csv is missing columns: {missing}")
        rows: dict[str, ParameterRow] = {}
        for r in table.itertuples():
            if r.name in rows:
                raise ParameterError(f"duplicate parameter row: {r.name}")
            row = ParameterRow(
                name=str(r.name),
                definition=_opt_str(r.definition),
                value=float(r.value),
                units=_opt_str(r.units),
                population=_opt_str(r.population),
                geography=_opt_str(r.geography),
                study_year=_opt_str(r.study_year),
                currency_year=_opt_str(r.currency_year),
                source_url=_opt_str(r.source_url),
                distribution=_opt_str(r.uncertainty_distribution),
                lower=_opt_float(r.lower_bound),
                upper=_opt_float(r.upper_bound),
                evidence_grade=_opt_str(r.evidence_grade),
                transformation=_opt_str(r.transformation),
                assumption_flag=_opt_str(r.assumption_flag),
            )
            rows[row.name] = row
        return cls(rows, PriceIndex.load(price_path))

    def resolve(
        self,
        price_year: int,
        overrides: dict[str, float] | None = None,
        draw: dict[str, float] | None = None,
    ) -> "Params":
        """Produce the flat value map the model reads.

        ``draw`` maps a parameter name to a uniform in (0, 1); those parameters
        are sampled from their uncertainty distribution. ``overrides`` is applied
        afterwards and wins, which is how scenarios pin a value.
        """
        draw = draw or {}
        values: dict[str, float] = {}
        for name, row in self.rows.items():
            v = row.quantile(draw[name]) if name in draw else row.value
            if row.is_money and row.currency_year:
                v *= self.prices.factor(int(float(row.currency_year)), price_year)
            values[name] = v
        for name, v in (overrides or {}).items():
            if name not in self.rows:
                raise ParameterError(f"override names a parameter that does not exist: {name}")
            values[name] = float(v)
        return Params(values, self, price_year, set(overrides or {}), set(draw))

    def structural_audit(self) -> list[str]:
        """Problems that make the table unusable, as plain sentences."""
        problems: list[str] = []
        for name, row in self.rows.items():
            if not row.units:
                problems.append(f"{name}: no units")
            if not row.definition:
                problems.append(f"{name}: no definition")
            if row.assumption_flag not in ASSUMPTION_FLAGS:
                problems.append(f"{name}: assumption_flag {row.assumption_flag!r} is not one of {sorted(ASSUMPTION_FLAGS)}")
            if row.distribution not in DISTRIBUTIONS:
                problems.append(f"{name}: distribution {row.distribution!r} is not one of {sorted(DISTRIBUTIONS)}")
            if row.distribution != "fixed":
                if row.lower is None or row.upper is None:
                    problems.append(f"{name}: {row.distribution} without both bounds")
                elif not (row.lower <= row.value <= row.upper):
                    problems.append(f"{name}: value {row.value} outside [{row.lower}, {row.upper}]")
            if row.is_money and not row.currency_year:
                problems.append(f"{name}: a USD value with no currency_year cannot be deflated")
            if row.assumption_flag == "observed_local" and not row.source_url:
                problems.append(f"{name}: flagged observed_local with no source_url")
        return problems


@dataclass
class Params:
    """A resolved, price-adjusted value map that records what the model read."""

    values: dict[str, float]
    registry: Registry
    price_year: int
    overridden: set[str] = field(default_factory=set)
    sampled: set[str] = field(default_factory=set)
    accessed: set[str] = field(default_factory=set)

    def __getitem__(self, name: str) -> float:
        try:
            v = self.values[name]
        except KeyError as exc:
            raise ParameterError(
                f"the model asked for {name!r}, which is not in the evidence registry"
            ) from exc
        self.accessed.add(name)
        return v

    def get(self, name: str, default: float | None = None) -> float:
        if name not in self.values and default is not None:
            return default
        return self[name]

    def vector(self, prefix: str, n: int) -> list[float]:
        return [self[f"{prefix}{i}"] for i in range(n)]

    def with_overrides(self, overrides: dict[str, float]) -> "Params":
        """A copy with some values pinned. Price adjustment already happened."""
        new = Params(dict(self.values), self.registry, self.price_year,
                     set(self.overridden) | set(overrides), set(self.sampled))
        for k, v in overrides.items():
            if k not in self.registry.rows:
                raise ParameterError(f"override names a parameter that does not exist: {k}")
            new.values[k] = float(v)
        return new

    def flag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in sorted(self.accessed):
            flag = self.registry.rows[name].assumption_flag
            counts[flag] = counts.get(flag, 0) + 1
        return counts

    def hypothetical_inputs(self) -> list[str]:
        return [n for n in sorted(self.accessed)
                if self.registry.rows[n].assumption_flag == "hypothetical"]

    def unused(self) -> list[str]:
        return sorted(set(self.registry.rows) - self.accessed)
