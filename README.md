# DOPart

 python .\project\plot.py --stages 0 --no-comms-uniform --log-uniform --alpha-min 0 --alpha-max 2.5 --alpha-fixed --lower-bound 0.25 --upper-bound 2.5

## Refactor Summary (2026-02-10)

### Benchmark command
`python .\project\plot.py --stages 0 --no-comms-uniform --log-uniform --alpha-min -1 --alpha-max 2.5 --alpha-fixed --lower-bound 0.25 --upper-bound 2.5`

### Performance
- Before: `583.067s` (from `profile.pstats`)
- After: `21.977s` (timed run with same command)
- Speedup: ~`26.5x`

### Main improvements
- Replaced repeated `sum(slice)` patterns with cached prefix/suffix cumulative sums in `project/methods.py`.
- Vectorized sample generation and makespan computation in `project/plot.py` using NumPy matrix operations.
- Removed redundant per-sample recomputation and reused randomized-threshold parameters.
- Skipped computing algorithms that are ignored downstream (`IGNORED_ALGS = {0, 2, 6, 8}`).
- Disabled seaborn bootstrap CI computation for faster plotting.
- Added robust figure-save fallback to local directory when Dropbox path is unavailable.

### Full documentation
- See `project/CodexRefactor` for complete change details, profiling notes, and reproducibility commands.
