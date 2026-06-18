"""Evaluation helpers for DOPart experiments."""

from __future__ import annotations

import random

import numpy as np

from .algorithms import ALPHAOPT, DOPart, DOPartR, DSR, rand_threshold_params
from .sampling import algorithm_bounds, sample_alphas, sample_communications

DEFAULT_ALGORITHMS = [
  "DOPart",
  "DOPart-R",
  "DSR",
  "Always Offload",
  "Never Offload",
  "OPT",
]


def evaluate_beta_setting(
  *,
  remote: np.ndarray,
  input_sizes: np.ndarray,
  samples: int,
  seed: int,
  alpha_min: float,
  alpha_max: float,
  communication_range_factor: float,
  beta_concentration: float,
  log_uniform: bool = True,
  random_min: bool = True,
  label: str | None = None,
) -> list[dict[str, float | int | str]]:
  rng = np.random.default_rng(seed)
  random.seed(seed)

  remote = np.asarray(remote, dtype=float)
  input_sizes = np.asarray(input_sizes, dtype=float)
  n_stages = remote.size
  base_bandwidth = float(input_sizes[0] / np.sum(remote))

  alpha_scales = sample_alphas(
    rng,
    alpha_min,
    alpha_max,
    log_uniform,
    size=(samples, n_stages),
  )
  local_samples = alpha_scales * remote
  comms_body = sample_communications(
    rng,
    input_sizes,
    base_bandwidth,
    communication_range_factor,
    samples,
    beta_concentration,
  )
  comms_samples = np.concatenate(
    (comms_body, np.zeros((samples, 1), dtype=float)),
    axis=1,
  )

  a, b = algorithm_bounds(alpha_min, alpha_max, log_uniform, random_min)
  dsr_params = rand_threshold_params(min(a, 1.0), max(b, 1.0))
  remote_total = float(np.sum(remote))

  rows: list[dict[str, float | int | str]] = []
  for sample_idx in range(samples):
    comms = comms_samples[sample_idx]
    local = local_samples[sample_idx]

    opt_cost, opt_idx, _ = ALPHAOPT(comms, local, remote)
    dopart_cost, _, dopart_idx, _ = DOPart(comms, local, remote, a, b, 0)
    dopartr_cost, _, dopartr_idx = DOPartR(comms, local, remote, a, b)
    dsr_cost, _, dsr_idx = DSR(comms, local, remote, a, b, rand_params=dsr_params)
    always_cost = float(comms[0] + remote_total)
    never_cost = float(np.sum(local))

    for alg, cost, offload_idx in (
      ("DOPart", dopart_cost, dopart_idx),
      ("DOPart-R", dopartr_cost, dopartr_idx),
      ("DSR", dsr_cost, dsr_idx),
      ("Always Offload", always_cost, 0),
      ("Never Offload", never_cost, n_stages),
      ("OPT", opt_cost, opt_idx),
    ):
      rows.append(
        {
          "Setting": label or "beta setting",
          "Sample": sample_idx,
          "Alg": alg,
          "Average Makespan": float(cost),
          "Regret": float(cost - opt_cost),
          "Competitive Ratio": float(cost / opt_cost) if opt_cost > 0 else np.inf,
          "Offload Layer": float(offload_idx),
          "Alpha Min": alpha_min,
          "Alpha Max": alpha_max,
          "Communication Range Factor": communication_range_factor,
          "Beta Concentration": beta_concentration,
        }
      )

  return rows


__all__ = ["DEFAULT_ALGORITHMS", "evaluate_beta_setting"]
