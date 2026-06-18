"""Small plotting helpers shared by experiment scripts."""

from __future__ import annotations

DEFAULT_ALGORITHM_ORDER = [
  "DOPart",
  "DOPart-R",
  "DSR",
  "Always Offload",
  "Never Offload",
  "OPT",
]


def lineplot_with_ci(sns, ax, lineplot_kwargs, ci_level: float):
  try:
    return sns.lineplot(ax=ax, errorbar=("ci", ci_level), **lineplot_kwargs)
  except TypeError:
    return sns.lineplot(ax=ax, ci=ci_level, **lineplot_kwargs)


def style_axis(ax, grid_alpha: float = 0.35):
  ax.grid(True, alpha=grid_alpha)
  for spine in ax.spines.values():
    spine.set_linewidth(0.5)


__all__ = ["DEFAULT_ALGORITHM_ORDER", "lineplot_with_ci", "style_axis"]
