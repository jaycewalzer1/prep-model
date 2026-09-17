"""State encodings shared by the deterministic and microsimulation models.

Overlapping dimensions are kept separate on purpose. Advanced disease and being
on ART can coexist, so ``HivStage`` and ``CareState`` are independent axes and
no person is ever forced into a mutually exclusive category that makes a real
clinical situation unrepresentable.
"""

from __future__ import annotations

from enum import IntEnum


class Housing(IntEnum):
    UNSHELTERED = 0
    SHELTERED = 1
    HOUSED = 2


class HivStage(IntEnum):
    SUSCEPTIBLE = 0
    ACUTE = 1
    CHRONIC = 2
    ADVANCED = 3


class CareState(IntEnum):
    """Only meaningful for people with HIV. Susceptible people sit at UNDIAGNOSED."""

    UNDIAGNOSED = 0
    DIAGNOSED_NO_ART = 1
    ART_UNSUPPRESSED = 2
    ART_SUPPRESSED = 3


class Product(IntEnum):
    NONE = 0
    ORAL = 1
    LENACAPAVIR = 2
    CABOTEGRAVIR = 3


PRODUCT_BY_NAME = {
    "none": Product.NONE,
    "oral": Product.ORAL,
    "lenacapavir": Product.LENACAPAVIR,
    "cabotegravir": Product.CABOTEGRAVIR,
}

COST_CATEGORIES = [
    "program_drug",
    "program_delivery",
    "program_testing",
    "hiv_care",
    "non_hiv_care",
    "housing",
    "patient_time",
]
CATEGORY_INDEX = {c: i for i, c in enumerate(COST_CATEGORIES)}

N_EXPOSURE_GROUPS = 4
