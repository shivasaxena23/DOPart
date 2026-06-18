from __future__ import annotations

import argparse
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

from data_generation import PROFILE_MODELS, system_values
from methods import ALPHAOPT, DOPart, DOPartR, DSR, rand_threshold_params

pd = None
plt = None
sns = None


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


def algorithm_bounds(alpha_min: float, alpha_max: float, log_uniform: bool, random_min: bool):
  if log_uniform:
    a = math.pow(2.0, alpha_min)
    b = math.pow(2.0, alpha_max)
  else:
    a = alpha_min
    b = alpha_max
  if random_min:
    a = min(a, 1.0)
  return a, b


def sample_communications(
  rng: np.random.Generator,
  input_sizes: np.ndarray,
  base_bandwidth: float,
  spread: float,
  samples: int,
  beta_concentration: float,
) -> np.ndarray:
  if spread < 1:
    raise ValueError("Communication range factor must be >= 1.")
  if beta_concentration <= 0:
    raise ValueError("Beta concentration must be positive.")
  if spread == 1:
    bandwidths = np.full((samples, input_sizes.size), base_bandwidth, dtype=float)
  else:
    lower = base_bandwidth / spread
    upper = base_bandwidth * spread
    mean_position = (base_bandwidth - lower) / (upper - lower)
    alpha = beta_concentration * mean_position
    beta = beta_concentration * (1.0 - mean_position)
    unit_samples = rng.beta(alpha, beta, size=(samples, input_sizes.size))
    bandwidths = lower + (upper - lower) * unit_samples
  return input_sizes / bandwidths


def evaluate_one_spread(
  rng: np.random.Generator,
  samples: int,
  spread: float,
  x_value: float,
  x_col: str,
  alpha_min: float,
  alpha_max: float,
  log_uniform: bool,
  random_min: bool,
  remote: np.ndarray,
  input_sizes: np.ndarray,
  beta_concentration: float,
):
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
    spread,
    samples,
    beta_concentration,
  )
  comms_samples = np.concatenate(
    (comms_body, np.zeros((samples, 1), dtype=float)),
    axis=1,
  )

  a, b = algorithm_bounds(alpha_min, alpha_max, log_uniform, random_min)
  dsr_params = rand_threshold_params(min(a, 1.0), max(b, 1.0))

  rows = []
  for sample_idx in range(samples):
    comms = comms_samples[sample_idx]
    local = local_samples[sample_idx]

    dopart_cost, _, dopart_idx, _ = DOPart(comms, local, remote, a, b, 0)
    dopartr_cost, _, dopartr_idx = DOPartR(comms, local, remote, a, b)
    dsr_cost, _, dsr_idx = DSR(comms, local, remote, a, b, rand_params=dsr_params)
    opt_cost, opt_idx, _ = ALPHAOPT(comms, local, remote)
    always_offload_cost = float(comms[0] + np.sum(remote))
    never_offload_cost = float(np.sum(local))

    for alg, cost, offload_idx in (
      ("DOPart", dopart_cost, dopart_idx),
      ("DOPart-R", dopartr_cost, dopartr_idx),
      ("DSR", dsr_cost, dsr_idx),
      ("Always Offload", always_offload_cost, 0),
      ("Never Offload", never_offload_cost, n_stages),
      ("OPT", opt_cost, opt_idx),
    ):
      rows.append(
        {
          x_col: x_value,
          "Average Makespan": cost,
          "Offload Layer": float(offload_idx),
          "Alg": alg,
        }
      )
  return rows


def generate_sweep(args):
  rng = np.random.default_rng(args.seed)
  random.seed(args.seed)

  remote, input_sizes = system_values(args.stages, profile_model=args.profile_model)
  remote = np.asarray(remote, dtype=float)
  input_sizes = np.asarray(input_sizes, dtype=float)
  if args.sweep == "range":
    x_values = build_sweep_values(args.range_min, args.range_max, args.range_period)
    x_col = "Communication Range Factor"
    x_label = "communication range factor"
  else:
    x_values = build_sweep_values(args.beta_min, args.beta_max, args.beta_period)
    x_col = "Beta Concentration"
    x_label = "beta concentration"

  rows = []
  for x_value in x_values:
    if args.sweep == "range":
      spread = x_value
      beta_concentration = args.beta_concentration
    else:
      spread = args.range_factor
      beta_concentration = x_value

    print(f"{x_label}: {x_value}")
    rows.extend(
      evaluate_one_spread(
        rng=rng,
        samples=args.samples,
        spread=spread,
        x_value=x_value,
        x_col=x_col,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        log_uniform=args.log_uniform,
        random_min=args.random_min,
        remote=remote,
        input_sizes=input_sizes,
        beta_concentration=beta_concentration,
      )
    )

  return pd.DataFrame(rows), x_values, remote.size, x_col, x_label


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


def build_plots(
  df,
  x_values: list[float],
  n_stages: int,
  ci_level: float,
  seed: int | None,
  x_col: str,
  x_label: str,
):
  active_algs = [
    "DOPart",
    "DOPart-R",
    "DSR",
    "Always Offload",
    "Never Offload",
    "OPT",
  ]
  d_style = {alg: "" for alg in active_algs}
  d_style["Always Offload"] = (2, 4)
  d_style["Never Offload"] = (2, 4)
  d_style["OPT"] = (5, 10)
  palette = sns.color_palette("tab10", n_colors=len(active_algs))

  sns.set_theme(font_scale=0.7, style="white")
  plt.rcParams["figure.figsize"] = [6, 3]
  plt.rcParams["figure.autolayout"] = True

  common_kwargs = dict(
    x=x_col,
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
  h_main.set_xlabel(x_label)
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
  h_offload.set_xlabel(x_label)
  h_offload.set_ylabel("Offload Layer Index")
  h_offload.set_ylim(-0.5, n_stages + 0.5)
  ytick_step = max(1, n_stages // 10)
  offload_ticks = np.arange(0, n_stages + 1, ytick_step, dtype=int)
  if offload_ticks[-1] != n_stages:
    offload_ticks = np.append(offload_ticks, n_stages)
  h_offload.set_yticks(offload_ticks)
  style_legend(ax_offload)
  ax_offload.grid(True)
  [spine.set_linewidth(0.5) for spine in ax_offload.spines.values()]

  return fig_main, fig_offload


def save_plots(figures, out_dir: Path, sweep: str):
  out_dir.mkdir(parents=True, exist_ok=True)
  ts = datetime.now().strftime("%Y-%b-%d_%H-%M-%S")
  stem = "DOPart_BetaConcentration" if sweep == "beta" else "DOPart_CommsRange"
  main_path = out_dir / f"{stem}_{ts}.pdf"
  offload_path = out_dir / f"{stem}_Offload_{ts}.pdf"
  figures[0].savefig(main_path, bbox_inches="tight")
  figures[1].savefig(offload_path, bbox_inches="tight")
  return main_path, offload_path


def print_summary(df, sweep_label: str):
  grouped = df.groupby("Alg", sort=False)
  print(f"\nOverall means across {sweep_label} sweep")
  print("--------------------------------" + "-" * len(sweep_label))
  for name in ["DOPart", "DOPart-R", "DSR", "Always Offload", "Never Offload", "OPT"]:
    mean_cost = float(grouped["Average Makespan"].mean()[name])
    mean_offload = float(grouped["Offload Layer"].mean()[name])
    print(f"{name:14s} mean makespan: {mean_cost:8.3f}   mean offload stage: {mean_offload:5.2f}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--samples", type=int, default=2500)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--stages", type=int, default=0)
  parser.add_argument("--profile-model", choices=PROFILE_MODELS, default="resnet34")
  parser.add_argument("--alpha-min", type=float, default=-1.0)
  parser.add_argument("--alpha-max", type=float, default=3.0)
  parser.add_argument("--log-uniform", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--random-min", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--sweep", choices=("range", "beta"), default="range")
  parser.add_argument(
    "--range-factor",
    type=float,
    default=20.0,
    help="Fixed communication range factor used when --sweep beta.",
  )
  parser.add_argument("--range-min", type=float, default=1.0)
  parser.add_argument("--range-max", type=float, default=20.0)
  parser.add_argument("--range-period", type=float, default=1.0)
  parser.add_argument(
    "--beta-concentration",
    type=float,
    default=20.0,
    help="Beta concentration for bandwidth sampling; mean bandwidth remains the baseline.",
  )
  parser.add_argument("--beta-min", type=float, default=2.0)
  parser.add_argument("--beta-max", type=float, default=80.0)
  parser.add_argument("--beta-period", type=float, default=2.0)
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

  df, x_values, n_stages, x_col, x_label = generate_sweep(args)
  print_summary(df, x_label)
  figures = build_plots(df, x_values, n_stages, args.ci, args.seed, x_col, x_label)

  if args.save:
    saved_paths = save_plots(figures, args.out_dir, args.sweep)
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
