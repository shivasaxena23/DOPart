# DNN Pipeline (Batch Offloading with DOPart)

## Purpose
`dnn_pipeline.py` extends the single-job DOPart decision model to a batch of `N` inference jobs.

For each job, the script:
- chooses a DNN split index using `DOPart`
- converts the split into 3 stage times:
  - local processing time
  - communication time
  - remote processing time
- schedules all jobs through a 3-resource pipeline

It then reports makespan and compares DOPart against an offline optimal solution.

## Resource Constraints
The scheduler enforces all three non-overlap constraints:
1. No two jobs process locally at the same time.
2. No two jobs communicate at the same time.
3. No two jobs process remotely at the same time.

Each job also obeys stage precedence:
`local -> communication -> remote`.

## Optimization Comparisons
The script prints three makespan values:
- `DOPart FIFO makespan`: jobs executed in input order with DOPart-selected splits.
- `DOPart best offline schedule makespan`: same DOPart splits, but best job ordering found by MILP.
- `Offline optimal makespan`: best joint split + schedule found by MILP (benchmark lower bound).

Reported ratios:
- `DOPart FIFO / Offline OPT`
- `DOPart best offline schedule / Offline OPT`

## Main Workflow
1. Load remote layer profile and activation sizes from `project/data_generation.py`.
2. Sample per-job local layer times via alpha scaling.
3. Sample communication times (bandwidth-based or uniform mode).
4. Build per-job split options:
   - local prefix time
   - communication at split
   - remote suffix time
5. Pick one split per job using `project/methods.py:DOPart`.
6. Simulate constrained 3-stage pipeline.
7. Solve offline MILP schedule(s) for comparison.

## Usage
From repository root:

```powershell
python "DNN Pipelining\dnn_pipeline.py" --num-jobs 5 --seed 1 --time-limit 120
```

Optional detailed output:

```powershell
python "DNN Pipelining\dnn_pipeline.py" --num-jobs 5 --print-jobs
```

## Key CLI Arguments
- `--num-jobs`: batch size `N`
- `--stage`: profile selector for `system_values()`
- `--seed`: random seed for repeatable sampling
- `--alpha-min`, `--alpha-max`, `--log-uniform`: local/remote scaling sampling
- `--comms-uniform`, `--lower-bound`, `--upper-bound`: communication sampling mode
- `--time-limit`: MILP solver time limit (seconds)
- `--print-jobs`: per-job split and stage durations

## Notes
- Offline optimization uses `scipy.optimize.milp` and can take longer for larger `N`.
- If the MILP time limit is too small, offline optimal may return `iteration_or_time_limit`.
- Default settings are tuned for a practical runtime/accuracy tradeoff.
