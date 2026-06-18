from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from data_generation import PROFILE_MODELS, system_values
from methods import ALPHAOPT, DOPart, DOPartR, DSR, rand_threshold_params
from plot_comms_range_sweep import (
  algorithm_bounds,
  build_sweep_values,
  sample_alphas,
  sample_communications,
)

pd = None
plt = None
sns = None

ALG_ORDER = [
  "DOPart",
  "DOPart-R",
  "DSR",
  "Always Offload",
  "Never Offload",
  "OPT",
]
COMPARISON_ALGS = [alg for alg in ALG_ORDER if alg != "OPT"]


@dataclass(frozen=True)
class Scenario:
  key: str
  title: str
  alpha_min: float
  alpha_max: float
  range_factor: float
  beta_concentration: float
  description: str


SCENARIOS = [
  Scenario(
    key="stable_control",
    title="Stable Communication",
    alpha_min=-1.0,
    alpha_max=3.0,
    range_factor=2.0,
    beta_concentration=80.0,
    description=(
      "Control case: bandwidth is tightly concentrated around the same mean. "
      "DOPart should be difficult to beat because the threshold decision is stable."
    ),
  ),
  Scenario(
    key="volatile_advantage",
    title="Volatile Hybrid Link",
    alpha_min=0.0,
    alpha_max=4.0,
    range_factor=3.0,
    beta_concentration=0.2,
    description=(
      "DOPart-R case: bandwidth follows a U-shaped beta distribution around the "
      "same mean, while local compute ranges from equal to slower than remote. "
      "Always-offload and never-offload are both poor, so this is a clean hybrid "
      "partitioning setting where randomized thresholding can help."
    ),
  ),
  Scenario(
    key="wide_range_boundary",
    title="Wider-Range Boundary",
    alpha_min=0.0,
    alpha_max=4.0,
    range_factor=10.0,
    beta_concentration=1.0,
    description=(
      "Boundary case: the communication range is wider and the beta distribution "
      "is less concentrated. This shows that the DOPart-R claim should be tied "
      "to the right regime, not stated as a universal dominance claim."
    ),
  ),
]


def evaluate_setting(
  scenario: Scenario,
  remote: np.ndarray,
  input_sizes: np.ndarray,
  samples: int,
  seed: int,
  log_uniform: bool,
  random_min: bool,
  beta_concentration: float | None = None,
  beta_label: float | None = None,
):
  rng = np.random.default_rng(seed)
  random.seed(seed)

  beta_value = scenario.beta_concentration if beta_concentration is None else beta_concentration
  beta_label_value = beta_value if beta_label is None else beta_label
  n_stages = remote.size
  base_bandwidth = float(input_sizes[0] / np.sum(remote))

  alpha_scales = sample_alphas(
    rng,
    scenario.alpha_min,
    scenario.alpha_max,
    log_uniform,
    size=(samples, n_stages),
  )
  local_samples = alpha_scales * remote
  comms_body = sample_communications(
    rng,
    input_sizes,
    base_bandwidth,
    scenario.range_factor,
    samples,
    beta_value,
  )
  comms_samples = np.concatenate(
    (comms_body, np.zeros((samples, 1), dtype=float)),
    axis=1,
  )

  a, b = algorithm_bounds(
    scenario.alpha_min,
    scenario.alpha_max,
    log_uniform,
    random_min,
  )
  dsr_params = rand_threshold_params(min(a, 1.0), max(b, 1.0))

  rows = []
  remote_total = float(np.sum(remote))
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
          "Scenario": scenario.title,
          "Scenario Key": scenario.key,
          "Sample": sample_idx,
          "Alg": alg,
          "Average Makespan": float(cost),
          "Regret": float(cost - opt_cost),
          "Competitive Ratio": float(cost / opt_cost) if opt_cost > 0 else np.inf,
          "Offload Layer": float(offload_idx),
          "Alpha Min": scenario.alpha_min,
          "Alpha Max": scenario.alpha_max,
          "Communication Range Factor": scenario.range_factor,
          "Beta Concentration": beta_label_value,
        }
      )

  return pd.DataFrame(rows)


def build_case_study_data(args, remote: np.ndarray, input_sizes: np.ndarray):
  frames = []
  for scenario_idx, scenario in enumerate(SCENARIOS):
    frames.append(
      evaluate_setting(
        scenario,
        remote,
        input_sizes,
        args.samples,
        args.seed + 1000 * scenario_idx,
        args.log_uniform,
        args.random_min,
      )
    )
  return pd.concat(frames, ignore_index=True)


def build_beta_sweep_data(args, remote: np.ndarray, input_sizes: np.ndarray):
  focus = next(scenario for scenario in SCENARIOS if scenario.key == "volatile_advantage")
  beta_values = args.beta_values
  if beta_values is None:
    beta_values = build_sweep_values(args.beta_min, args.beta_max, args.beta_period)

  frames = []
  for beta_idx, beta_concentration in enumerate(beta_values):
    print(f"beta concentration: {beta_concentration}")
    frames.append(
      evaluate_setting(
        focus,
        remote,
        input_sizes,
        args.samples,
        args.seed + 10000 + beta_idx,
        args.log_uniform,
        args.random_min,
        beta_concentration=float(beta_concentration),
        beta_label=float(beta_concentration),
      )
    )
  return pd.concat(frames, ignore_index=True), [float(value) for value in beta_values]


def lineplot_with_ci(ax, lineplot_kwargs, ci_level: float):
  try:
    return sns.lineplot(ax=ax, errorbar=("ci", ci_level), **lineplot_kwargs)
  except TypeError:
    return sns.lineplot(ax=ax, ci=ci_level, **lineplot_kwargs)


def barplot_with_ci(ax, barplot_kwargs, ci_level: float):
  try:
    return sns.barplot(ax=ax, errorbar=("ci", ci_level), **barplot_kwargs)
  except TypeError:
    fallback_kwargs = dict(barplot_kwargs)
    fallback_kwargs.pop("legend", None)
    return sns.barplot(ax=ax, ci=ci_level, **fallback_kwargs)


def style_axis(ax):
  ax.grid(True, alpha=0.35)
  for spine in ax.spines.values():
    spine.set_linewidth(0.5)


def save_figure(fig, out_dir: Path, filename: str, save: bool):
  saved_path = None
  if save:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_path = out_dir / filename
    fig.savefig(saved_path, bbox_inches="tight")
  return saved_path


def plot_beta_sweep(df, beta_values: list[float], args):
  plot_algs = ["DOPart", "DOPart-R", "DSR", "Always Offload", "Never Offload"]
  plot_df = df[df["Alg"].isin(plot_algs)]

  fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
  common = dict(
    x="Beta Concentration",
    hue="Alg",
    style="Alg",
    hue_order=plot_algs,
    style_order=plot_algs,
    data=plot_df,
    linewidth=1.2,
    markers=True,
    markersize=6,
    seed=args.seed,
  )

  lineplot_with_ci(axes[0], dict(common, y="Regret"), args.ci)
  axes[0].set_title("Mean Regret")
  axes[0].set_xlabel("Beta concentration")
  axes[0].set_ylabel(r"$T_\mathregular{ALG} - T_\mathregular{OPT}$")
  axes[0].set_xscale("log")
  axes[0].set_xticks(beta_values)
  axes[0].set_xticklabels([f"{value:g}" for value in beta_values])
  style_axis(axes[0])

  lineplot_with_ci(axes[1], dict(common, y="Offload Layer"), args.ci)
  axes[1].set_title("Mean Offload Layer")
  axes[1].set_xlabel("Beta concentration")
  axes[1].set_ylabel("Offload layer index")
  axes[1].set_xscale("log")
  axes[1].set_xticks(beta_values)
  axes[1].set_xticklabels([f"{value:g}" for value in beta_values])
  style_axis(axes[1])

  handles, labels = axes[1].get_legend_handles_labels()
  axes[0].get_legend().remove()
  axes[1].get_legend().remove()
  fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
  fig.suptitle("DOPart-R Advantage Under Volatile Bandwidth")
  fig.tight_layout(rect=[0, 0, 1, 0.88])
  return fig


def plot_scenario_regret(df, args):
  fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(12, 3.2), sharey=True)
  palette = sns.color_palette("tab10", n_colors=len(COMPARISON_ALGS))

  for ax, scenario in zip(axes, SCENARIOS):
    plot_df = df[
      (df["Scenario Key"] == scenario.key)
      & (df["Alg"].isin(COMPARISON_ALGS))
    ]
    barplot_with_ci(
      ax,
      dict(
        data=plot_df,
        x="Alg",
        hue="Alg",
        y="Regret",
        order=COMPARISON_ALGS,
        hue_order=COMPARISON_ALGS,
        palette=palette,
        legend=False,
        seed=args.seed,
      ),
      args.ci,
    )
    ax.set_title(scenario.title)
    ax.set_xlabel("")
    ax.set_ylabel("Mean regret" if ax is axes[0] else "")
    ax.tick_params(axis="x", rotation=35)
    style_axis(ax)

  fig.suptitle("Regret Over OPT By Scenario")
  fig.tight_layout(rect=[0, 0, 1, 0.9])
  return fig


def plot_opt_offload_histograms(df, n_stages: int):
  fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(12, 3.2), sharey=True)
  bins = np.arange(-0.5, n_stages + 1.5, 1)
  line_colors = {"DOPart": "tab:blue", "DOPart-R": "tab:orange"}

  for ax, scenario in zip(axes, SCENARIOS):
    scenario_df = df[df["Scenario Key"] == scenario.key]
    opt_layers = scenario_df[scenario_df["Alg"] == "OPT"]["Offload Layer"].to_numpy()
    weights = np.ones_like(opt_layers, dtype=float) / max(1, opt_layers.size)
    ax.hist(
      opt_layers,
      bins=bins,
      weights=weights,
      color="0.65",
      edgecolor="white",
      linewidth=0.5,
    )
    for alg, color in line_colors.items():
      mean_layer = float(
        scenario_df[scenario_df["Alg"] == alg]["Offload Layer"].mean()
      )
      ax.axvline(mean_layer, color=color, linewidth=1.5, label=f"{alg} mean")
    ax.set_title(scenario.title)
    ax.set_xlabel("Offload layer index")
    ax.set_xlim(-0.5, n_stages + 0.5)
    ax.set_ylabel("Probability" if ax is axes[0] else "")
    style_axis(ax)

  handles, labels = axes[-1].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
  fig.suptitle("Where OPT Offloads, With Algorithm Mean Decisions")
  fig.tight_layout(rect=[0, 0, 1, 0.88])
  return fig


def plot_regret_cdf(df):
  focus_df = df[
    (df["Scenario Key"] == "volatile_advantage")
    & (df["Alg"].isin(["DOPart", "DOPart-R", "DSR"]))
  ]
  fig, ax = plt.subplots(figsize=(6, 3.2))
  for alg in ["DOPart", "DOPart-R", "DSR"]:
    values = np.sort(focus_df[focus_df["Alg"] == alg]["Regret"].to_numpy())
    y = np.arange(1, values.size + 1, dtype=float) / values.size
    ax.plot(values, y, label=alg, linewidth=1.4)
  ax.set_title("Regret CDF In The Volatile-Wireless Case")
  ax.set_xlabel(r"$T_\mathregular{ALG} - T_\mathregular{OPT}$")
  ax.set_ylabel("Fraction of samples")
  ax.legend(frameon=False)
  style_axis(ax)
  fig.tight_layout()
  return fig


def print_summaries(df):
  print("\nScenario summaries")
  print("------------------")
  for scenario in SCENARIOS:
    scenario_df = df[df["Scenario Key"] == scenario.key]
    means = (
      scenario_df.groupby("Alg", sort=False)
      .agg(
        mean_makespan=("Average Makespan", "mean"),
        mean_regret=("Regret", "mean"),
        mean_offload=("Offload Layer", "mean"),
      )
      .reindex(ALG_ORDER)
    )
    dopart_regret = float(means.loc["DOPart", "mean_regret"])
    dopartr_regret = float(means.loc["DOPart-R", "mean_regret"])
    improvement = dopart_regret - dopartr_regret

    print(f"\n{scenario.title}")
    print(scenario.description)
    print(
      "alpha=[{:.1f}, {:.1f}], range_factor={:.1f}, beta_concentration={:.1f}".format(
        scenario.alpha_min,
        scenario.alpha_max,
        scenario.range_factor,
        scenario.beta_concentration,
      )
    )
    print(
      "DOPart-R regret improvement over DOPart: "
      f"{improvement:.3f} ({'better' if improvement > 0 else 'worse'})"
    )
    for alg in ALG_ORDER:
      row = means.loc[alg]
      print(
        f"{alg:14s} makespan={row['mean_makespan']:8.3f} "
        f"regret={row['mean_regret']:8.3f} "
        f"offload={row['mean_offload']:5.2f}"
      )


def print_beta_sweep_summary(df):
  means = (
    df.groupby(["Beta Concentration", "Alg"], sort=True)["Regret"]
    .mean()
    .unstack("Alg")
  )
  print("\nVolatile-link beta sweep")
  print("------------------------")
  print("Positive improvement means DOPart-R has lower regret than DOPart.")
  for beta_value, row in means.iterrows():
    improvement = float(row["DOPart"] - row["DOPart-R"])
    print(
      f"beta={beta_value:5g}  "
      f"DOPart regret={row['DOPart']:8.3f}  "
      f"DOPart-R regret={row['DOPart-R']:8.3f}  "
      f"improvement={improvement:8.3f}"
    )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--samples", type=int, default=2500)
  parser.add_argument("--seed", type=int, default=321)
  parser.add_argument("--stages", type=int, default=0)
  parser.add_argument("--profile-model", choices=PROFILE_MODELS, default="resnet34")
  parser.add_argument("--log-uniform", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--random-min", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--beta-min", type=float, default=0.2)
  parser.add_argument("--beta-max", type=float, default=80.0)
  parser.add_argument("--beta-period", type=float, default=0.2)
  parser.add_argument(
    "--beta-values",
    type=float,
    nargs="+",
    default=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 80.0],
    help="Explicit beta concentrations for the volatile-wireless sweep.",
  )
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

  sns.set_theme(font_scale=0.7, style="white")
  plt.rcParams["figure.autolayout"] = False

  remote, input_sizes = system_values(args.stages, profile_model=args.profile_model)
  remote = np.asarray(remote, dtype=float)
  input_sizes = np.asarray(input_sizes, dtype=float)

  case_df = build_case_study_data(args, remote, input_sizes)
  beta_df, beta_values = build_beta_sweep_data(args, remote, input_sizes)
  print_summaries(case_df)
  print_beta_sweep_summary(beta_df)

  figures = [
    (plot_beta_sweep(beta_df, beta_values, args), "dopartr_case_beta_sweep.pdf"),
    (plot_scenario_regret(case_df, args), "dopartr_case_scenario_regret.pdf"),
    (plot_opt_offload_histograms(case_df, remote.size), "dopartr_case_opt_offload_hist.pdf"),
    (plot_regret_cdf(case_df), "dopartr_case_regret_cdf.pdf"),
  ]

  saved_paths = [
    save_figure(fig, args.out_dir, filename, args.save)
    for fig, filename in figures
  ]
  if args.save:
    print("\nSaved plots:")
    for path in saved_paths:
      print(f"  {path}")

  if args.show:
    plt.show()
  else:
    for fig, _ in figures:
      plt.close(fig)


if __name__ == "__main__":
  main()
