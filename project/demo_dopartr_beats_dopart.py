"""Sweep-style synthetic plot where DOPart-R beats deterministic DOPart.

The figures intentionally follow the spirit of project/plot.py: average
makespan and offload-layer curves over an alpha-axis sweep.  The synthetic
workload is a two-cluster construction:

* Early-edge samples: OPT is in the first half, but the communication delay is
  just above DOPart's deterministic threshold. DOPart rejects these, while high
  randomized DOPart-R thresholds sometimes accept them.
* Local-only samples: OPT is the final stage because all communication delays
  are too expensive. Both DOPart and DOPart-R correctly stay local.

This demonstrates the situation where OPT is concentrated in the first half or
at the last stage, and DOPart-R wins by recovering part of the first cluster.
Communication delay is still computed as data size divided by sampled bandwidth;
the synthetic data sizes are chosen after sampling bandwidths to create the
desired threshold-edge cases.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib
import numpy as np

from methods import ALPHAOPT, DOPart, DOPartR, rand_threshold_params

pd = None
plt = None
sns = None


def build_sweep_values(start: float, end: float, step: float) -> list[float]:
  if step <= 0:
    raise ValueError("Sweep step must be > 0.")
  if end < start:
    raise ValueError("Sweep end must be >= sweep start.")
  return [round(start + step * i, 10) for i in range(int(round((end - start) / step)) + 1)]


def randomized_threshold_upper(r: float, R: float) -> float:
  ep, _ = rand_threshold_params(min(r, 1.0), max(R, 1.0))
  return max(R, 1.0) - ep


def dopart_comm_threshold(remote: np.ndarray, stage_idx: int, r: float, R: float) -> float:
  local = R * remote
  prefix = float(np.sum(local[:stage_idx]))
  suffix = float(np.sum(remote[stage_idx:]))
  current_remote = float(remote[stage_idx])
  suffix_next = float(np.sum(remote[stage_idx + 1:]))

  term0 = prefix + suffix
  term1 = prefix + r * current_remote + min(r, 1.0) * suffix_next
  term2 = prefix + max(R, 1.0) * suffix
  return max(0.0, math.sqrt(term1 * term2) - term0)


def dopartr_max_comm_threshold(remote: np.ndarray, stage_idx: int, r: float, R: float) -> float:
  local = R * remote
  prefix = float(np.sum(local[:stage_idx]))
  suffix = float(np.sum(remote[stage_idx:]))
  t_bar = prefix + suffix
  if t_bar == 0:
    return 0.0

  alpha_min = (prefix + min(r, 1.0) * suffix) / t_bar
  alpha_max = (prefix + max(R, 1.0) * suffix) / t_bar
  stage_upper = randomized_threshold_upper(alpha_min, alpha_max)
  return max(0.0, (stage_upper - 1.0) * t_bar)


def sample_bandwidths(
  rng: np.random.Generator,
  n_points: int,
  bandwidth_min: float,
  bandwidth_max: float,
) -> np.ndarray:
  if bandwidth_min <= 0 or bandwidth_max <= 0:
    raise ValueError("Bandwidth bounds must be positive.")
  if bandwidth_max < bandwidth_min:
    raise ValueError("bandwidth_max must be >= bandwidth_min.")

  log_min = math.log(bandwidth_min)
  log_max = math.log(bandwidth_max)
  return np.exp(rng.uniform(log_min, log_max, size=n_points))


def comms_from_bandwidths(data_sizes: np.ndarray, bandwidths: np.ndarray) -> np.ndarray:
  return data_sizes / bandwidths


def make_early_edge_instance(
  rng: np.random.Generator,
  n_stages: int,
  r: float,
  R: float,
  bandwidth_min: float,
  bandwidth_max: float,
):
  remote = rng.lognormal(mean=0.0, sigma=0.15, size=n_stages)
  local = R * remote
  remote_total = float(remote.sum())
  bandwidths = sample_bandwidths(rng, n_stages + 1, bandwidth_min, bandwidth_max)

  first_half_limit = max(1, n_stages // 2)
  stage_idx = int(rng.integers(0, first_half_limit))
  det_comm = dopart_comm_threshold(remote, stage_idx, r, R)
  dopartr_upper_comm = dopartr_max_comm_threshold(remote, stage_idx, r, R)

  if dopartr_upper_comm <= det_comm:
    stage_idx = 0
    det_comm = dopart_comm_threshold(remote, stage_idx, r, R)
    dopartr_upper_comm = dopartr_max_comm_threshold(remote, stage_idx, r, R)

  gap = max(dopartr_upper_comm - det_comm, 1e-6)
  target_comm = rng.uniform(det_comm + 0.18 * gap, det_comm + 0.82 * gap)

  comm_targets = np.full(n_stages + 1, 100.0 * remote_total, dtype=float)
  comm_targets[stage_idx] = min(target_comm, 0.90 * float(np.sum(local)))
  comm_targets[-1] = 0.0
  data_sizes = comm_targets * bandwidths
  comms = comms_from_bandwidths(data_sizes, bandwidths)
  return comms, local, remote, "Early OPT"


def make_local_only_instance(
  rng: np.random.Generator,
  n_stages: int,
  R: float,
  bandwidth_min: float,
  bandwidth_max: float,
):
  remote = rng.lognormal(mean=0.0, sigma=0.15, size=n_stages)
  local = R * remote
  remote_total = float(remote.sum())
  bandwidths = sample_bandwidths(rng, n_stages + 1, bandwidth_min, bandwidth_max)

  comm_targets = np.full(n_stages + 1, 100.0 * remote_total, dtype=float)
  comm_targets[-1] = 0.0
  data_sizes = comm_targets * bandwidths
  comms = comms_from_bandwidths(data_sizes, bandwidths)
  return comms, local, remote, "Local OPT"


def generate_sweep(
  samples: int,
  seed: int,
  n_stages: int,
  alpha_min: float,
  x_values: list[float],
  early_probability: float,
  bandwidth_min: float,
  bandwidth_max: float,
):
  rows = []
  rng = np.random.default_rng(seed)
  random.seed(seed)

  for alpha_max in x_values:
    for _ in range(samples):
      if rng.random() < early_probability:
        comms, local, remote, cluster = make_early_edge_instance(
          rng=rng,
          n_stages=n_stages,
          r=alpha_min,
          R=alpha_max,
          bandwidth_min=bandwidth_min,
          bandwidth_max=bandwidth_max,
        )
      else:
        comms, local, remote, cluster = make_local_only_instance(
          rng=rng,
          n_stages=n_stages,
          R=alpha_max,
          bandwidth_min=bandwidth_min,
          bandwidth_max=bandwidth_max,
        )

      dopart_cost, _, dopart_idx, _ = DOPart(comms, local, remote, alpha_min, alpha_max, 0)
      dopartr_cost, _, dopartr_idx = DOPartR(comms, local, remote, alpha_min, alpha_max)
      opt_cost, opt_idx, _ = ALPHAOPT(comms, local, remote)

      for alg, cost, offload_idx in (
        ("DOPart", dopart_cost, dopart_idx),
        ("DOPart-R", dopartr_cost, dopartr_idx),
        ("OPT", opt_cost, opt_idx),
      ):
        rows.append(
          {
            "Average Makespan": cost,
            "Offload Layer": float(offload_idx),
            "AlphaMax": alpha_max,
            "Alg": alg,
            "Cluster": cluster,
          }
        )

  return pd.DataFrame(rows)


def lineplot_with_ci(ax, lineplot_kwargs, ci_level: float):
  ci_enabled = ci_level > 0
  try:
    if ci_enabled:
      return sns.lineplot(ax=ax, errorbar=("ci", ci_level), **lineplot_kwargs)
    return sns.lineplot(ax=ax, errorbar=None, **lineplot_kwargs)
  except TypeError:
    if ci_enabled:
      return sns.lineplot(ax=ax, ci=ci_level, **lineplot_kwargs)
    return sns.lineplot(ax=ax, ci=None, **lineplot_kwargs)


def style_legend(ax):
  handles, labels = ax.get_legend_handles_labels()
  if len(handles) >= 2:
    handles[0], handles[1] = handles[1], handles[0]
    labels[0], labels[1] = labels[1], labels[0]
  ax.legend(handles=handles, labels=labels)


def build_plots(df, x_values: list[float], n_stages: int, ci_level: float, seed: int | None):
  active_algs = ["DOPart", "DOPart-R", "OPT"]
  d_style = {alg: "" for alg in active_algs}
  d_style["OPT"] = (5, 10)
  palette = sns.color_palette("tab10", n_colors=len(active_algs))

  sns.set_theme(font_scale=0.7, style="white")
  plt.rcParams["figure.figsize"] = [6, 3]
  plt.rcParams["figure.autolayout"] = True

  common_kwargs = dict(
    x="AlphaMax",
    hue="Alg",
    data=df,
    style="Alg",
    linewidth=1,
    hue_order=active_algs,
    style_order=active_algs,
    palette=palette,
    markers=True,
    dashes=d_style,
    markersize=8,
    seed=seed,
  )

  fig_main, ax_main = plt.subplots()
  h_main = lineplot_with_ci(
    ax_main,
    dict(common_kwargs, y="Average Makespan"),
    ci_level,
  )
  h_main.set_xticks(x_values)
  h_main.set_xlabel(r"$\alpha_\mathregular{max}$")
  h_main.set_ylabel(r"Average " + r"$T_\mathregular{ALG}$")
  h_main.ticklabel_format(useMathText=True)
  style_legend(ax_main)
  ax_main.grid(True)
  [spine.set_linewidth(0.5) for spine in ax_main.spines.values()]

  fig_offload, ax_offload = plt.subplots()
  h_offload = lineplot_with_ci(
    ax_offload,
    dict(common_kwargs, y="Offload Layer"),
    ci_level,
  )
  h_offload.set_xticks(x_values)
  h_offload.set_xlabel(r"$\alpha_\mathregular{max}$")
  h_offload.set_ylabel("Offload Layer Index")
  h_offload.set_ylim(-0.5, n_stages + 0.5)
  h_offload.set_yticks(range(n_stages + 1))
  style_legend(ax_offload)
  ax_offload.grid(True)
  [spine.set_linewidth(0.5) for spine in ax_offload.spines.values()]

  opt_df = df[df["Alg"] == "OPT"].copy()
  opt_df["OPT Region"] = np.where(
    opt_df["Offload Layer"] < n_stages / 2,
    "First half",
    "Last stage",
  )
  region_df = (
    opt_df
    .groupby(["AlphaMax", "OPT Region"], as_index=False)
    .size()
    .rename(columns={"size": "Count"})
  )
  region_df["Probability"] = region_df["Count"] / region_df.groupby("AlphaMax")["Count"].transform("sum")

  fig_region, ax_region = plt.subplots()
  h_region = sns.lineplot(
    ax=ax_region,
    x="AlphaMax",
    y="Probability",
    hue="OPT Region",
    style="OPT Region",
    data=region_df,
    markers=True,
    dashes=False,
    linewidth=1,
  )
  h_region.set_xticks(x_values)
  h_region.set_xlabel(r"$\alpha_\mathregular{max}$")
  h_region.set_ylabel("OPT Region Probability")
  h_region.set_ylim(-0.05, 1.05)
  ax_region.grid(True)
  [spine.set_linewidth(0.5) for spine in ax_region.spines.values()]

  return fig_main, fig_offload, fig_region


def save_plots(figures, out_dir: Path):
  out_dir.mkdir(parents=True, exist_ok=True)
  main_path = out_dir / "dopartr_beats_dopart_sweep_makespan.pdf"
  offload_path = out_dir / "dopartr_beats_dopart_sweep_offload.pdf"
  region_path = out_dir / "dopartr_beats_dopart_sweep_opt_regions.pdf"
  figures[0].savefig(main_path, bbox_inches="tight")
  figures[1].savefig(offload_path, bbox_inches="tight")
  figures[2].savefig(region_path, bbox_inches="tight")
  return main_path, offload_path, region_path


def print_summary(df):
  means = df.groupby("Alg", sort=False)["Average Makespan"].mean()
  offloads = df.groupby("Alg", sort=False)["Offload Layer"].mean()
  improvement = 100.0 * (means["DOPart"] - means["DOPart-R"]) / means["DOPart"]
  opt_df = df[df["Alg"] == "OPT"]
  first_half = float(np.mean(opt_df["Offload Layer"] < opt_df["Offload Layer"].max() / 2))
  last_stage = float(np.mean(opt_df["Offload Layer"] == opt_df["Offload Layer"].max()))

  print("Synthetic bimodal-OPT alpha_max sweep")
  print("-------------------------------------")
  for name in ["DOPart", "DOPart-R", "OPT"]:
    print(f"{name:8s} mean makespan: {means[name]:8.3f}   mean offload stage: {offloads[name]:5.2f}")
  print(f"\nOPT first-half share: {100.0 * first_half:.2f}%")
  print(f"OPT last-stage share: {100.0 * last_stage:.2f}%")
  print(f"\nDOPart-R improvement over DOPart: {improvement:.2f}%")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--samples", type=int, default=2500)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--stages", type=int, default=8)
  parser.add_argument("--alpha-min", type=float, default=1.0)
  parser.add_argument("--alpha-max-min", type=float, default=2.0)
  parser.add_argument("--alpha-max-max", type=float, default=10.0)
  parser.add_argument("--period", type=float, default=1.0)
  parser.add_argument("--early-probability", type=float, default=0.65)
  parser.add_argument("--bandwidth-min", type=float, default=1.0)
  parser.add_argument("--bandwidth-max", type=float, default=1000.0)
  parser.add_argument("--ci", type=float, default=95.0)
  parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--save", action="store_true")
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path(__file__).resolve().parent / "plots",
  )
  args = parser.parse_args()

  global pd, plt, sns
  if not args.show:
    matplotlib.use("Agg", force=True)
  import matplotlib.pyplot as pyplot
  import pandas as pandas
  import seaborn as seaborn
  pd = pandas
  plt = pyplot
  sns = seaborn

  x_values = build_sweep_values(args.alpha_max_min, args.alpha_max_max, args.period)
  df = generate_sweep(
    samples=args.samples,
    seed=args.seed,
    n_stages=args.stages,
    alpha_min=args.alpha_min,
    x_values=x_values,
    early_probability=args.early_probability,
    bandwidth_min=args.bandwidth_min,
    bandwidth_max=args.bandwidth_max,
  )
  print_summary(df)

  figures = build_plots(df, x_values, args.stages, args.ci, args.seed)

  if args.save:
    saved_paths = save_plots(figures, args.out_dir)
    print("\nSaved plots:")
    for path in saved_paths:
      print(f"  {path}")

  if args.show:
    plt.show()
  else:
    for figure in figures:
      plt.close(figure)


if __name__ == "__main__":
  main()
