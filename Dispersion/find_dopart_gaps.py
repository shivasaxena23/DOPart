from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from dopart.algorithms import ALPHAOPT, DOPart
from dopart.profiles import PROFILE_MODELS, system_values
from dopart.sampling import algorithm_bounds, sample_alphas, sample_communications


@dataclass(frozen=True)
class Scenario:
  profile: str
  alpha_min: float
  alpha_max: float
  range_factor: float
  beta_concentration: float


DEFAULT_ALPHA_RANGES = [
  (-4.0, 2.0),
  (-2.0, 2.0),
  (-1.0, 3.0),
  (0.0, 3.0),
  (0.0, 4.0),
  (1.0, 4.0),
]

CSV_FIELDS = [
  "rank",
  "profile",
  "alpha_min",
  "alpha_max",
  "range_factor",
  "beta_concentration",
  "samples_per_seed",
  "seeds",
  "dopart_mean",
  "opt_mean",
  "dopart_regret",
  "dopart_ratio",
  "best_theta",
  "best_theta_mean",
  "best_theta_regret",
  "learnable_improvement",
  "learnable_improvement_pct_of_gap",
  "always_offload_mean",
  "never_offload_mean",
  "best_endpoint_mean",
  "endpoint_margin",
  "opt_first_frac",
  "opt_last_frac",
  "opt_interior_frac",
  "opt_unique_layers",
  "dopart_mean_offload",
  "opt_mean_offload",
  "scope_score",
]


def parse_float_list(values: list[str] | None) -> list[float] | None:
  if values is None:
    return None
  return [float(value) for value in values]


def parse_alpha_ranges(values: list[str] | None) -> list[tuple[float, float]]:
  if not values:
    return DEFAULT_ALPHA_RANGES
  ranges = []
  for value in values:
    if ":" not in value:
      raise ValueError(f"Alpha range must look like MIN:MAX, got {value!r}.")
    start, end = value.split(":", 1)
    ranges.append((float(start), float(end)))
  return ranges


def dopart_scaled_threshold(
  comms: np.ndarray,
  local: np.ndarray,
  remote: np.ndarray,
  alm: float,
  alM: float,
  theta: float,
) -> tuple[float, int]:
  local_prefix = np.empty(local.size + 1, dtype=float)
  local_prefix[0] = 0.0
  if local.size:
    np.cumsum(local, out=local_prefix[1:])

  remote_suffix = np.empty(remote.size + 1, dtype=float)
  remote_suffix[-1] = 0.0
  if remote.size:
    remote_suffix[:-1] = np.cumsum(remote[::-1])[::-1]

  best_local = float(local_prefix[-1])
  min_alm = min(alm, 1.0)
  max_alM = max(alM, 1.0)
  span = min(comms.size, local_prefix.size, remote_suffix.size)

  for i in range(span):
    prefix_i = local_prefix[i]
    suffix_i = remote_suffix[i]
    pr_i = remote[i] if i < remote.size else 0.0
    suffix_next = remote_suffix[i + 1] if (i + 1) < remote_suffix.size else 0.0

    term0 = prefix_i + suffix_i
    term1 = prefix_i + alm * pr_i + min_alm * suffix_next
    term2 = prefix_i + max_alM * suffix_i
    threshold = max(0.0, math.sqrt(term1 * term2) - term0)

    if comms[i] <= theta * threshold:
      return float(term0 + comms[i]), i

  return best_local, int(comms.size)


def sample_scenario(
  scenario: Scenario,
  samples: int,
  seed: int,
  log_uniform: bool,
  random_min: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
  with contextlib.redirect_stdout(io.StringIO()):
    remote, input_sizes = system_values(0, profile_model=scenario.profile)
  remote = np.asarray(remote, dtype=float)
  input_sizes = np.asarray(input_sizes, dtype=float)
  rng = np.random.default_rng(seed)

  alpha_scales = sample_alphas(
    rng,
    scenario.alpha_min,
    scenario.alpha_max,
    log_uniform,
    size=(samples, remote.size),
  )
  local_samples = alpha_scales * remote
  base_bandwidth = float(input_sizes[0] / np.sum(remote))
  comms_body = sample_communications(
    rng,
    input_sizes,
    base_bandwidth,
    scenario.range_factor,
    samples,
    scenario.beta_concentration,
  )
  comms_samples = np.concatenate(
    (comms_body, np.zeros((samples, 1), dtype=float)),
    axis=1,
  )
  alm, alM = algorithm_bounds(
    scenario.alpha_min,
    scenario.alpha_max,
    log_uniform,
    random_min,
  )
  return remote, local_samples, comms_samples, alm, alM


def evaluate_scenario(
  scenario: Scenario,
  samples: int,
  seeds: int,
  seed_offset: int,
  theta_grid: np.ndarray,
  log_uniform: bool,
  random_min: bool,
) -> dict[str, float | int | str]:
  dopart_costs = []
  opt_costs = []
  always_costs = []
  never_costs = []
  dopart_layers = []
  opt_layers = []
  theta_cost_sums = np.zeros(theta_grid.size, dtype=float)
  total_samples = 0
  n_stages = 0

  for seed_idx in range(seeds):
    seed = seed_offset + 1000003 * seed_idx
    remote, local_samples, comms_samples, alm, alM = sample_scenario(
      scenario,
      samples,
      seed,
      log_uniform,
      random_min,
    )
    n_stages = remote.size
    remote_total = float(np.sum(remote))

    for sample_idx in range(samples):
      local = local_samples[sample_idx]
      comms = comms_samples[sample_idx]
      opt_cost, opt_idx, _ = ALPHAOPT(comms, local, remote)
      dopart_cost, _, dopart_idx, _ = DOPart(comms, local, remote, alm, alM, 0)

      dopart_costs.append(dopart_cost)
      opt_costs.append(opt_cost)
      always_costs.append(float(comms[0] + remote_total))
      never_costs.append(float(np.sum(local)))
      dopart_layers.append(float(dopart_idx))
      opt_layers.append(float(opt_idx))

      for theta_idx, theta in enumerate(theta_grid):
        theta_cost, _ = dopart_scaled_threshold(
          comms,
          local,
          remote,
          alm,
          alM,
          float(theta),
        )
        theta_cost_sums[theta_idx] += theta_cost

      total_samples += 1

  dopart_arr = np.asarray(dopart_costs, dtype=float)
  opt_arr = np.asarray(opt_costs, dtype=float)
  always_arr = np.asarray(always_costs, dtype=float)
  never_arr = np.asarray(never_costs, dtype=float)
  opt_layer_arr = np.asarray(opt_layers, dtype=float)
  dopart_layer_arr = np.asarray(dopart_layers, dtype=float)

  theta_means = theta_cost_sums / max(1, total_samples)
  best_theta_idx = int(np.argmin(theta_means))
  best_theta = float(theta_grid[best_theta_idx])
  best_theta_mean = float(theta_means[best_theta_idx])

  dopart_mean = float(np.mean(dopart_arr))
  opt_mean = float(np.mean(opt_arr))
  dopart_regret = dopart_mean - opt_mean
  best_theta_regret = best_theta_mean - opt_mean
  learnable_improvement = dopart_mean - best_theta_mean
  best_endpoint_mean = float(min(np.mean(always_arr), np.mean(never_arr)))
  endpoint_margin = best_endpoint_mean - best_theta_mean
  opt_first_frac = float(np.mean(opt_layer_arr == 0))
  opt_last_frac = float(np.mean(opt_layer_arr == n_stages))
  opt_interior_frac = 1.0 - opt_first_frac - opt_last_frac

  # Score scenarios high when DOPart has a large gap, a tuned threshold closes
  # some of it, and endpoint policies do not already solve the scenario.
  endpoint_bonus = max(0.0, endpoint_margin)
  scope_score = max(0.0, learnable_improvement) * math.log1p(max(0.0, dopart_regret))
  scope_score += 0.1 * endpoint_bonus

  return {
    "profile": scenario.profile,
    "alpha_min": scenario.alpha_min,
    "alpha_max": scenario.alpha_max,
    "range_factor": scenario.range_factor,
    "beta_concentration": scenario.beta_concentration,
    "samples_per_seed": samples,
    "seeds": seeds,
    "dopart_mean": dopart_mean,
    "opt_mean": opt_mean,
    "dopart_regret": dopart_regret,
    "dopart_ratio": float(np.mean(dopart_arr / opt_arr)),
    "best_theta": best_theta,
    "best_theta_mean": best_theta_mean,
    "best_theta_regret": best_theta_regret,
    "learnable_improvement": learnable_improvement,
    "learnable_improvement_pct_of_gap": (
      learnable_improvement / dopart_regret if dopart_regret > 0 else 0.0
    ),
    "always_offload_mean": float(np.mean(always_arr)),
    "never_offload_mean": float(np.mean(never_arr)),
    "best_endpoint_mean": best_endpoint_mean,
    "endpoint_margin": endpoint_margin,
    "opt_first_frac": opt_first_frac,
    "opt_last_frac": opt_last_frac,
    "opt_interior_frac": opt_interior_frac,
    "opt_unique_layers": int(np.unique(opt_layer_arr).size),
    "dopart_mean_offload": float(np.mean(dopart_layer_arr)),
    "opt_mean_offload": float(np.mean(opt_layer_arr)),
    "scope_score": scope_score,
  }


def build_scenarios(
  profiles: list[str],
  alpha_ranges: list[tuple[float, float]],
  range_factors: list[float],
  beta_concentrations: list[float],
) -> list[Scenario]:
  scenarios = []
  for profile in profiles:
    for alpha_min, alpha_max in alpha_ranges:
      for range_factor in range_factors:
        for beta_concentration in beta_concentrations:
          scenarios.append(
            Scenario(
              profile=profile,
              alpha_min=alpha_min,
              alpha_max=alpha_max,
              range_factor=range_factor,
              beta_concentration=beta_concentration,
            )
          )
  return scenarios


def write_csv(path: Path, rows: list[dict[str, float | int | str]]):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def print_top(rows: list[dict[str, float | int | str]], top_k: int):
  print("\nTop DOPart learning-scope scenarios")
  print("-----------------------------------")
  for row in rows[:top_k]:
    print(
      "#{rank:02d} {profile:18s} "
      "alpha=[{alpha_min:g},{alpha_max:g}] "
      "range={range_factor:g} beta={beta_concentration:g} "
      "gap={dopart_regret:.3f} "
      "theta={best_theta:.3f} "
      "improve={learnable_improvement:.3f} "
      "improve_pct={learnable_improvement_pct_of_gap:.2%} "
      "endpoint_margin={endpoint_margin:.3f} "
      "OPT_int={opt_interior_frac:.2%}".format(**row)
    )


def main():
  parser = argparse.ArgumentParser(
    description="Find scenario distributions where DOPart is far from OPT.",
  )
  parser.add_argument("--profiles", nargs="+", default=["resnet34"], choices=PROFILE_MODELS)
  parser.add_argument(
    "--alpha-ranges",
    nargs="+",
    default=None,
    help="Alpha ranges as MIN:MAX, e.g. --alpha-ranges -1:3 0:4.",
  )
  parser.add_argument(
    "--range-factors",
    nargs="+",
    type=float,
    default=[2.0, 3.0, 5.0, 10.0, 20.0],
  )
  parser.add_argument(
    "--beta-concentrations",
    nargs="+",
    type=float,
    default=[0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 80.0],
  )
  parser.add_argument("--samples", type=int, default=400)
  parser.add_argument("--seeds", type=int, default=2)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--theta-min", type=float, default=0.4)
  parser.add_argument("--theta-max", type=float, default=2.5)
  parser.add_argument("--theta-count", type=int, default=43)
  parser.add_argument("--log-uniform", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--random-min", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--top-k", type=int, default=20)
  parser.add_argument(
    "--out",
    type=Path,
    default=Path(__file__).resolve().parent / "results" / "dopart_gap_scenarios.csv",
  )
  args = parser.parse_args()

  if args.theta_count < 2:
    raise ValueError("--theta-count must be at least 2.")

  alpha_ranges = parse_alpha_ranges(args.alpha_ranges)
  theta_grid = np.linspace(args.theta_min, args.theta_max, args.theta_count)
  scenarios = build_scenarios(
    args.profiles,
    alpha_ranges,
    args.range_factors,
    args.beta_concentrations,
  )

  print(f"Scanning {len(scenarios)} scenarios")
  print(f"Theta grid: {args.theta_min:g} to {args.theta_max:g} ({args.theta_count} points)")

  rows = []
  for scenario_idx, scenario in enumerate(scenarios, start=1):
    print(
      f"[{scenario_idx}/{len(scenarios)}] "
      f"{scenario.profile} alpha=[{scenario.alpha_min:g},{scenario.alpha_max:g}] "
      f"range={scenario.range_factor:g} beta={scenario.beta_concentration:g}"
    )
    row = evaluate_scenario(
      scenario,
      samples=args.samples,
      seeds=args.seeds,
      seed_offset=args.seed + 1009 * scenario_idx,
      theta_grid=theta_grid,
      log_uniform=args.log_uniform,
      random_min=args.random_min,
    )
    rows.append(row)

  rows.sort(
    key=lambda row: (
      float(row["scope_score"]),
      float(row["dopart_regret"]),
      float(row["learnable_improvement"]),
    ),
    reverse=True,
  )
  for rank, row in enumerate(rows, start=1):
    row["rank"] = rank

  write_csv(args.out, rows)
  print_top(rows, args.top_k)
  print(f"\nWrote results: {args.out}")


if __name__ == "__main__":
  main()
