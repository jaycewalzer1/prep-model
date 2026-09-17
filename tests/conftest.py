"""Shared fixtures. Tests run a small, short model so the whole suite is quick;
the checks are about invariants, which do not need a large cohort to hold."""

from __future__ import annotations

import pytest

from prep_model.config import load_config
from prep_model.params import Registry


@pytest.fixture(scope="session")
def registry() -> Registry:
    return Registry.load()


@pytest.fixture(scope="session")
def cfg():
    """A deliberately small configuration: 800 people for 2 years."""
    return load_config("config/base.yaml").copy_with(
        run_name="test",
        population={"n_individuals": 800},
        time={"horizon_years": 2},
        uncertainty={"stochastic_replicates": 1, "parameter_draws": 0},
    )


@pytest.fixture(scope="session")
def params(registry, cfg):
    return registry.resolve(cfg.price_year, overrides=cfg.overrides)
