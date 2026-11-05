"""Whole-epoch cumulative-prefix schedules shared by SISA and ProtoSISA."""
from math import gcd


def epochs_per_slice(epochs, slices):
    """Return 2E/(R+1), rejecting schedules that would round or skip stages.

    E is the nominal full-shard epoch budget and R the number of slices.
    Valid integer E values are multiples of (R+1)/gcd(2, R+1).
    """
    for name, value in (("epochs", epochs), ("slices", slices)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    passes, remainder = divmod(2 * epochs, slices + 1)
    if remainder or passes < 1:
        multiple = (slices + 1) // gcd(2, slices + 1)
        raise ValueError(
            f"Fractional epochs per slice are not allowed: 2 * {epochs} / "
            f"({slices} + 1) must be a positive integer; "
            f"epochs must be a positive multiple of {multiple} for {slices} slices"
        )
    return passes
