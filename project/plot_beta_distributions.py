from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

plt = None


def beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
  import math

  log_norm = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
  interior = (x > 0) & (x < 1)
  y = np.zeros_like(x, dtype=float)
  y[interior] = np.exp(
    log_norm
    + (alpha - 1.0) * np.log(x[interior])
    + (beta - 1.0) * np.log(1.0 - x[interior])
  )
  return y


def beta_params_for_mean(range_factor: float, concentration: float):
  if range_factor < 1:
    raise ValueError("range_factor must be >= 1.")
  if concentration <= 0:
    raise ValueError("concentration must be > 0.")
  if range_factor == 1:
    return None

  lower = 1.0 / range_factor
  upper = range_factor
  mean_position = (1.0 - lower) / (upper - lower)
  alpha = concentration * mean_position
  beta = concentration * (1.0 - mean_position)
  return alpha, beta, lower, upper


def plot_beta_distributions(range_factor: float, concentrations: list[float], out: Path | None):
  params = beta_params_for_mean(range_factor, concentrations[0])
  if params is None:
    raise ValueError("range_factor=1 gives a degenerate fixed-bandwidth distribution.")
  _, _, lower, upper = params

  unit_x = np.linspace(0.001, 0.999, 1200)
  multiplier_x = lower + (upper - lower) * unit_x

  fig, ax = plt.subplots(figsize=(6, 3.2))
  for concentration in concentrations:
    alpha, beta, _, _ = beta_params_for_mean(range_factor, concentration)
    density = beta_pdf(unit_x, alpha, beta) / (upper - lower)
    ax.plot(
      multiplier_x,
      density,
      linewidth=1.4,
      label=rf"$\kappa={concentration:g}$",
    )

  ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="mean")
  ax.set_xscale("log")
  ax.set_xlabel("bandwidth multiplier")
  ax.set_ylabel("density")
  ax.set_title(rf"Beta bandwidth distributions, range factor={range_factor:g}")
  ax.grid(True, alpha=0.3)
  ax.legend()
  fig.tight_layout()

  if out is not None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved plot: {out}")

  return fig


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--range-factor", type=float, default=20.0)
  parser.add_argument("--concentrations", type=float, nargs="+", default=[2.0, 20.0, 80.0])
  parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--save", action="store_true")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path(__file__).resolve().parent / "plots" / "beta_bandwidth_distributions.pdf",
  )
  args = parser.parse_args()

  global plt
  if not args.show:
    matplotlib.use("Agg", force=True)
  import matplotlib.pyplot as pyplot
  plt = pyplot

  fig = plot_beta_distributions(
    range_factor=args.range_factor,
    concentrations=args.concentrations,
    out=args.out if args.save else None,
  )

  if args.show:
    plt.show()
  else:
    plt.close(fig)


if __name__ == "__main__":
  main()
