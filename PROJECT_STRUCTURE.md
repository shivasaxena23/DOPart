# Project Structure

This repository is organized around two layers:

- `dopart/`: reusable code for algorithms, profile loading, sampling,
  evaluation, and plotting helpers.
- `experiments/`: reproducible scripts for paper figures and exploratory runs.

The older `project/` directory is still kept as a compatibility layer. Existing
commands such as `python .\project\plot.py ...` should continue to work.

## Recommended Entry Points

```powershell
python .\experiments\alpha_sweep.py
python .\experiments\comms_range_sweep.py --sweep beta --range-factor 3
python .\experiments\beta_distributions.py --range-factor 3
python .\experiments\dopartr_case_studies.py --save --no-show
```

## Directory Guide

```text
dopart/
  algorithms.py      stable imports for DOPart, DOPart-R, DSR, OPT
  profiles.py        profile loading
  sampling.py        alpha and communication sampling
  evaluation.py      makespan, regret, offload-index rows
  plotting.py        shared plotting helpers

experiments/
  alpha_sweep.py
  comms_range_sweep.py
  beta_distributions.py
  dopartr_case_studies.py
  synthetic_dopartr_demo.py

data/
  profiles/          future home for profile text files
  raw/               raw measurements or large inputs
  raw/legacy/        older measurements kept for reproducibility
  profiling_scripts/ scripts used to create model profiles

outputs/
  plots/             generated plots
  tables/            generated CSV/LaTeX tables
  logs/              run logs
  profiles/          profiler outputs such as .pstats

paper/
  references/        paper PDFs and source references
  notes/             derivation notes and investigation logs
  figures/           final selected figures for manuscripts

notebooks/
  exploratory Jupyter notebooks

theory/
  online_search/     one-way trading and online-search scratch/proof code
```

## Suggested Migration Path

1. Keep using `project/` scripts until the paper figures are stable.
2. Move shared logic from `project/plot*.py` into `dopart/evaluation.py`,
   `dopart/sampling.py`, and `dopart/plotting.py` as experiments mature.
3. Put new figure scripts in `experiments/` first.
4. Treat `outputs/` as disposable generated output.

## Where Loose Files Went

- Root PDFs and generated plots moved to `outputs/plots/archive/`.
- The v32 paper PDF moved to `paper/references/`.
- Investigation notes moved to `paper/notes/`.
- Jupyter notebooks moved to `notebooks/`.
- One-way trading / online-search scripts moved to `theory/online_search/`.
- Legacy raw profile measurements moved to `data/raw/legacy/`.
- Profiler outputs moved to `outputs/profiles/`.
- The DNN pipelining side project moved under `experiments/pipelining/`.
