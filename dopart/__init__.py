"""Reusable DOPart package.

The package exposes the core algorithms and helpers under stable import paths
while the older scripts in ``project/`` remain available as compatibility
entrypoints.
"""

from .algorithms import (
  ALPHAOPT,
  DOPart,
  DOPartARAND,
  DOPartR,
  DOPartRAND,
  DSR,
  rand_threshold_params,
)
from .profiles import PROFILE_MODELS, system_values

__all__ = [
  "ALPHAOPT",
  "DOPart",
  "DOPartARAND",
  "DOPartR",
  "DOPartRAND",
  "DSR",
  "PROFILE_MODELS",
  "rand_threshold_params",
  "system_values",
]
