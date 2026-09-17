"""Counter-based random draws.

Paired policy runs need the *same* random number for the same person, the same
week and the same event type, whatever else differs between arms. Resetting a
seed does not achieve that: as soon as one arm's branching consumes a different
number of draws, every later stream slides out of alignment.

So draws here are a pure function of ``(seed, stream, step, person_uid)``. No
generator state is carried anywhere. Two arms that ask for the infection draw
of person 8123 in week 41 get the identical uniform, even if one arm reached
that week having run twice as much code.

The hash is splitmix64 evaluated in int64 tensors. PyTorch's integer arithmetic
wraps, which is what splitmix64 wants; the only care needed is that ``>>`` on a
signed tensor is an arithmetic shift, so logical shifts are masked by hand.
"""

from __future__ import annotations

from enum import IntEnum

import torch

_MASK64 = (1 << 64) - 1


def _signed(x: int) -> int:
    x &= _MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


_GOLDEN = _signed(0x9E3779B97F4A7C15)
_MIX1 = _signed(0xBF58476D1CE4E5B9)
_MIX2 = _signed(0x94D049BB133111EB)
_TWO_POW_M53 = 2.0**-53


class Stream(IntEnum):
    """One identifier per event type. Never reuse a value for two events."""

    ENTRY_COUNT = 1
    ATTR_AGE = 2
    ATTR_HOUSING = 3
    ATTR_EXPOSURE = 4
    ATTR_INJECTS = 5
    ATTR_HIV = 6
    ATTR_CARE = 7
    ATTR_ORAL_PREP = 8
    ATTR_ENGAGEMENT = 9
    EXIT = 10
    EXIT_CAUSE = 11
    INFECTION = 12
    ACUTE_PROGRESSION = 13
    DISEASE_PROGRESSION = 14
    TESTING = 15
    ART_INIT = 16
    SUPPRESSION = 17
    ART_FAILURE = 18
    DISENGAGE = 19
    REENGAGE = 20
    HOUSING_MOVE = 21
    HOUSING_DEST = 22
    PREP_REACH = 23
    PREP_ACCEPT = 24
    PREP_INITIATE = 25
    PREP_ATTEND = 26
    PREP_REENGAGE = 27
    ART_REGRESSION = 28
    ATTR_STAGE = 29
    ATTR_TIME_SINCE_INFECTION = 30
    ATTR_ART_DELAY = 31
    DEATH_ATTRIBUTION = 32
    PROGRAM_TEST = 33
    PREP_ADVERSE_EVENT = 34
    PREP_POST_STOP_ORAL = 35
    PREP_REFUSAL_PERMANENT = 36
    PROGRAM_LINKAGE = 37


def _lsr(x: torch.Tensor, k: int) -> torch.Tensor:
    """Logical (unsigned) right shift of a signed int64 tensor."""
    return (x >> k) & ((1 << (64 - k)) - 1)


def _splitmix_scalar(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _splitmix(z: torch.Tensor) -> torch.Tensor:
    z = z + _GOLDEN
    z = (z ^ _lsr(z, 30)) * _MIX1
    z = (z ^ _lsr(z, 27)) * _MIX2
    return z ^ _lsr(z, 31)


def stream_key(seed: int, stream: Stream | int, step: int) -> int:
    """Collapse the scalar part of the coordinate into one int64 key."""
    k = _splitmix_scalar(seed & _MASK64)
    k = _splitmix_scalar(k ^ (int(stream) * 0x9E3779B9))
    k = _splitmix_scalar(k ^ (int(step) * 0xD1B54A32D192ED03))
    return _signed(k)


def uniform(seed: int, stream: Stream | int, step: int, uid: torch.Tensor) -> torch.Tensor:
    """Uniform(0, 1) draws, one per entry of ``uid``.

    Deterministic in ``(seed, stream, step, uid)`` alone.
    """
    key = stream_key(seed, stream, step)
    h = _splitmix(_splitmix(uid.to(torch.int64) + key))
    return _lsr(h, 11).to(torch.float64) * _TWO_POW_M53


def normal(seed: int, stream: Stream | int, step: int, uid: torch.Tensor) -> torch.Tensor:
    """Standard normal draws by Box-Muller on two independent uniform streams."""
    u1 = uniform(seed, stream, step, uid).clamp_min(1e-12)
    u2 = uniform(seed, int(stream) + 1000, step, uid)
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2 * torch.pi * u2)


def categorical(u: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF categorical sampling from one uniform per row.

    ``probs`` is ``(n, k)`` and each row must sum to at most 1; any residual
    mass maps to the last category, so callers should pass normalised rows.
    """
    cum = torch.cumsum(probs, dim=1)
    return (u.unsqueeze(1) > cum).sum(dim=1).clamp_max(probs.shape[1] - 1)
