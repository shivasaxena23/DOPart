Plot inconsistency checkup (updated):

Issue observed:
- `python .\project\plot.py --stages 0 --no-comms-uniform --log-uniform --alpha-min -5 --alpha-max 5 --no-alpha-fixed --lower-bound 0.05 --upper-bound 2.5 --random-min --ci 95`
- Results varied across runs.

Root cause:
- Multiple random sources were active (`numpy`, Python `random`, randomized methods, and CI bootstrap in seaborn).

Changes made:
- Added `--ci` support in `project/plot.py` for confidence intervals.
- Added `--seed` in `project/plot.py` for reproducible runs.
- `--seed` now sets both `np.random.seed(seed)` and `random.seed(seed)`.
- Seaborn lineplot now also receives `seed=seed` so CI bootstrapping is reproducible.
- Updated `project/findCommsRange.py` sampling to option 2:
  - draw `x,y ~ Uniform(0.0001, 5)` independently
  - set `(l,u) = sorted((x,y))` to enforce `l <= u`

Use for reproducible output:
- `python .\project\plot.py --stages 0 --no-comms-uniform --log-uniform --alpha-min -5 --alpha-max 5 --no-alpha-fixed --lower-bound 0.05 --upper-bound 2.5 --random-min --ci 95 --seed 42`

Note:
- Without `--seed`, run-to-run variability is expected.
