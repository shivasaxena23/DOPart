"""Sampling utilities shared by experiments."""

from __future__ import annotations

import math

import numpy as np


def build_sweep_values(start: float, end: float, step: float) -> list[float]:
  if step <= 0:
    raise ValueError("Sweep step must be > 0.")
  if end < start:
    raise ValueError("Sweep end must be >= sweep start.")
  return [round(start + step * i, 10) for i in range(int(round((end - start) / step)) + 1)]


def sample_alphas(
  rng: np.random.Generator,
  alpha_min: float,
  alpha_max: float,
  log_uniform: bool,
  size: tuple[int, int],
) -> np.ndarray:
  if log_uniform:
    return np.power(2.0, rng.uniform(alpha_min, alpha_max, size=size))
  return rng.uniform(alpha_min, alpha_max, size=size)


def algorithm_bounds(
  alpha_min: float,
  alpha_max: float,
  log_uniform: bool,
  random_min: bool,
) -> tuple[float, float]:
  if log_uniform:
    a = math.pow(2.0, alpha_min)
    b = math.pow(2.0, alpha_max)
  else:
    a = alpha_min
    b = alpha_max
  if random_min:
    a = min(a, 1.0)
  return a, b


def beta_bandwidths(
  rng: np.random.Generator,
  base_bandwidth: float,
  spread: float,
  beta_concentration: float,
  size: tuple[int, int],
) -> np.ndarray:
  if spread < 1:
    raise ValueError("Communication range factor must be >= 1.")
  if beta_concentration <= 0:
    raise ValueError("Beta concentration must be positive.")
  if spread == 1:
    return np.full(size, base_bandwidth, dtype=float)

  lower = base_bandwidth / spread
  upper = base_bandwidth * spread
  mean_position = (base_bandwidth - lower) / (upper - lower)
  alpha = beta_concentration * mean_position
  beta = beta_concentration * (1.0 - mean_position)
  unit_samples = rng.beta(alpha, beta, size=size)
  return lower + (upper - lower) * unit_samples


def sample_communications(
  rng: np.random.Generator,
  input_sizes: np.ndarray,
  base_bandwidth: float,
  spread: float,
  samples: int,
  beta_concentration: float,
) -> np.ndarray:
  bandwidths = beta_bandwidths(
    rng,
    base_bandwidth,
    spread,
    beta_concentration,
    size=(samples, input_sizes.size),
  )
  return input_sizes / bandwidths


__all__ = [
  "algorithm_bounds",
  "beta_bandwidths",
  "build_sweep_values",
  "sample_alphas",
  "sample_communications",
]
