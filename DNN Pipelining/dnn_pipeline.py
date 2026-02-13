import argparse
import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_generation import system_values  # noqa: E402
from methods import DOPart  # noqa: E402


@dataclass
class ScheduleResult:
    makespan: float
    order: np.ndarray
    local_start: np.ndarray
    local_end: np.ndarray
    comm_start: np.ndarray
    comm_end: np.ndarray
    remote_start: np.ndarray
    remote_end: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline N DNN jobs across local/communication/remote resources with strict "
            "non-overlap, then compare DOPart against offline optimal."
        )
    )
    parser.add_argument("--num-jobs", type=int, default=5, help="Number of jobs in the batch.")
    parser.add_argument("--stage", type=int, default=0, help="Profile stage index used by system_values().")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument("--alpha-min", type=float, default=0.25, help="Lower alpha sampling bound.")
    parser.add_argument("--alpha-max", type=float, default=2.5, help="Upper alpha sampling bound.")
    parser.add_argument("--log-uniform", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--comms-uniform", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lower-bound", type=float, default=0.25, help="Bandwidth lower factor.")
    parser.add_argument("--upper-bound", type=float, default=2.5, help="Bandwidth upper factor.")

    parser.add_argument(
        "--time-limit",
        type=float,
        default=120.0,
        help="MILP solver time limit in seconds for each optimization run.",
    )
    parser.add_argument(
        "--print-jobs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-job split and stage-time details.",
    )
    return parser.parse_args()


def build_job_samples(
    num_jobs: int,
    stage: int,
    alpha_min: float,
    alpha_max: float,
    log_uniform: bool,
    comms_uniform: bool,
    lower_bound: float,
    upper_bound: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if num_jobs <= 0:
        raise ValueError("--num-jobs must be positive.")
    if alpha_max < alpha_min:
        raise ValueError("--alpha-max must be >= --alpha-min.")
    if upper_bound < lower_bound:
        raise ValueError("--upper-bound must be >= --lower-bound.")

    rng = np.random.default_rng(seed)

    with contextlib.redirect_stdout(io.StringIO()):
        remote_layers, input_data_real = system_values(stage)
    remote_layers = np.asarray(remote_layers, dtype=float).reshape(-1)
    input_data_real = np.asarray(input_data_real, dtype=float).reshape(-1)

    if remote_layers.size == 0:
        raise ValueError("system_values() returned an empty remote profile.")
    if input_data_real.size != remote_layers.size:
        raise ValueError(
            "Input-size profile length does not match remote profile length: "
            f"{input_data_real.size} vs {remote_layers.size}."
        )

    if log_uniform:
        r = float(2.0 ** alpha_min)
        R = float(2.0 ** alpha_max)
        alpha_scales = np.power(2.0, rng.uniform(alpha_min, alpha_max, size=(num_jobs, remote_layers.size)))
    else:
        r = float(alpha_min)
        R = float(alpha_max)
        alpha_scales = rng.uniform(alpha_min, alpha_max, size=(num_jobs, remote_layers.size))

    local_layers = alpha_scales * remote_layers
    total_remote = float(np.sum(remote_layers))

    if comms_uniform:
        comms_body = (r - 1.0 + rng.random((num_jobs, remote_layers.size)) * (R - r)) * total_remote
    else:
        bandwidth = input_data_real[0] / total_remote
        sampled_bw = (
            bandwidth * lower_bound
            + rng.random((num_jobs, remote_layers.size)) * bandwidth * upper_bound
        )
        comms_body = input_data_real / sampled_bw

    comms = np.concatenate((comms_body, np.zeros((num_jobs, 1), dtype=float)), axis=1)
    return remote_layers, local_layers, comms, r, R


def build_stage_options(
    local_layers: np.ndarray,
    remote_layers: np.ndarray,
    comms: np.ndarray,
) -> np.ndarray:
    num_jobs, num_layers = local_layers.shape
    num_splits = num_layers + 1

    local_prefix = np.cumsum(local_layers, axis=1, dtype=float)
    local_prefix = np.concatenate((np.zeros((num_jobs, 1), dtype=float), local_prefix), axis=1)

    remote_suffix = np.empty(num_splits, dtype=float)
    remote_suffix[-1] = 0.0
    if num_layers > 0:
        remote_suffix[:-1] = np.cumsum(remote_layers[::-1], dtype=float)[::-1]

    if comms.shape != (num_jobs, num_splits):
        raise ValueError(f"Expected comms shape {(num_jobs, num_splits)} but got {comms.shape}.")

    options = np.empty((num_jobs, num_splits, 3), dtype=float)
    options[:, :, 0] = local_prefix
    options[:, :, 1] = comms
    options[:, :, 2] = remote_suffix[None, :]
    return options


def pick_dopart_splits(
    local_layers: np.ndarray,
    remote_layers: np.ndarray,
    comms: np.ndarray,
    r: float,
    R: float,
) -> np.ndarray:
    num_jobs = local_layers.shape[0]
    max_valid_split = comms.shape[1] - 1
    splits = np.zeros(num_jobs, dtype=int)

    for j in range(num_jobs):
        _, _, raw_split, _ = DOPart(comms[j], local_layers[j], remote_layers, r, R, 0)
        split = int(raw_split)
        split = max(0, min(split, max_valid_split))
        splits[j] = split

    return splits


def stage_times_from_splits(stage_options: np.ndarray, splits: np.ndarray) -> np.ndarray:
    return stage_options[np.arange(stage_options.shape[0]), splits]


def simulate_pipeline(stage_times: np.ndarray, order: np.ndarray | None = None) -> ScheduleResult:
    num_jobs = stage_times.shape[0]
    if order is None:
        order = np.arange(num_jobs, dtype=int)
    else:
        order = np.asarray(order, dtype=int)

    local_start = np.zeros(num_jobs, dtype=float)
    local_end = np.zeros(num_jobs, dtype=float)
    comm_start = np.zeros(num_jobs, dtype=float)
    comm_end = np.zeros(num_jobs, dtype=float)
    remote_start = np.zeros(num_jobs, dtype=float)
    remote_end = np.zeros(num_jobs, dtype=float)

    local_free = 0.0
    comm_free = 0.0
    remote_free = 0.0

    for job in order:
        p_local, p_comm, p_remote = stage_times[job]

        l_start = local_free
        l_end = l_start + p_local
        c_start = max(comm_free, l_end)
        c_end = c_start + p_comm
        r_start = max(remote_free, c_end)
        r_end = r_start + p_remote

        local_start[job] = l_start
        local_end[job] = l_end
        comm_start[job] = c_start
        comm_end[job] = c_end
        remote_start[job] = r_start
        remote_end[job] = r_end

        local_free = l_end
        comm_free = c_end
        remote_free = r_end

    return ScheduleResult(
        makespan=float(remote_free),
        order=order,
        local_start=local_start,
        local_end=local_end,
        comm_start=comm_start,
        comm_end=comm_end,
        remote_start=remote_start,
        remote_end=remote_end,
    )


def solve_optimal_schedule(
    stage_options: np.ndarray,
    time_limit_s: float,
) -> tuple[bool, str, float | None, np.ndarray | None, np.ndarray | None]:
    num_jobs, num_splits, _ = stage_options.shape

    x_idx = np.arange(num_jobs * num_splits).reshape(num_jobs, num_splits)
    next_idx = num_jobs * num_splits

    local_idx = np.arange(next_idx, next_idx + num_jobs)
    next_idx += num_jobs
    comm_idx = np.arange(next_idx, next_idx + num_jobs)
    next_idx += num_jobs
    remote_idx = np.arange(next_idx, next_idx + num_jobs)
    next_idx += num_jobs

    y_idx: dict[tuple[int, int, int], int] = {}
    for machine in range(3):
        for j in range(num_jobs):
            for k in range(j + 1, num_jobs):
                y_idx[(machine, j, k)] = next_idx
                next_idx += 1

    cmax_idx = next_idx
    n_vars = cmax_idx + 1

    c = np.zeros(n_vars, dtype=float)
    c[cmax_idx] = 1.0

    lb = np.zeros(n_vars, dtype=float)
    ub = np.full(n_vars, np.inf, dtype=float)

    integrality = np.zeros(n_vars, dtype=np.int8)
    integrality[x_idx.reshape(-1)] = 1
    for idx in y_idx.values():
        integrality[idx] = 1
        ub[idx] = 1.0
    ub[x_idx.reshape(-1)] = 1.0

    # Conservative makespan upper bound for Big-M constraints.
    horizon = float(np.sum(np.max(np.sum(stage_options, axis=2), axis=1)))
    if horizon <= 0.0:
        horizon = 1.0

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coeffs: list[tuple[int, float]], lb_row: float, ub_row: float) -> None:
        row_id = len(lower)
        for col, val in coeffs:
            if val == 0.0:
                continue
            rows.append(row_id)
            cols.append(col)
            vals.append(float(val))
        lower.append(float(lb_row))
        upper.append(float(ub_row))

    # Each job picks exactly one split.
    for j in range(num_jobs):
        add_constraint([(int(x_idx[j, s]), 1.0) for s in range(num_splits)], 1.0, 1.0)

    # Precedence constraints: Local -> Comm -> Remote for each job.
    for j in range(num_jobs):
        add_constraint(
            [(int(local_idx[j]), 1.0), (int(comm_idx[j]), -1.0)]
            + [(int(x_idx[j, s]), float(stage_options[j, s, 0])) for s in range(num_splits)],
            -np.inf,
            0.0,
        )
        add_constraint(
            [(int(comm_idx[j]), 1.0), (int(remote_idx[j]), -1.0)]
            + [(int(x_idx[j, s]), float(stage_options[j, s, 1])) for s in range(num_splits)],
            -np.inf,
            0.0,
        )

    machine_starts = [local_idx, comm_idx, remote_idx]

    # Disjunctive non-overlap on each resource.
    for machine in range(3):
        starts = machine_starts[machine]
        for j in range(num_jobs):
            for k in range(j + 1, num_jobs):
                y = y_idx[(machine, j, k)]

                # start_j + p_j <= start_k + M * (1 - y)
                add_constraint(
                    [
                        (int(starts[j]), 1.0),
                        (int(starts[k]), -1.0),
                        (int(y), horizon),
                    ]
                    + [(int(x_idx[j, s]), float(stage_options[j, s, machine])) for s in range(num_splits)],
                    -np.inf,
                    horizon,
                )

                # start_k + p_k <= start_j + M * y
                add_constraint(
                    [
                        (int(starts[k]), 1.0),
                        (int(starts[j]), -1.0),
                        (int(y), -horizon),
                    ]
                    + [(int(x_idx[k, s]), float(stage_options[k, s, machine])) for s in range(num_splits)],
                    -np.inf,
                    0.0,
                )

    # Makespan definition.
    for j in range(num_jobs):
        add_constraint(
            [(int(remote_idx[j]), 1.0), (int(cmax_idx), -1.0)]
            + [(int(x_idx[j, s]), float(stage_options[j, s, 2])) for s in range(num_splits)],
            -np.inf,
            0.0,
        )

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(len(lower), n_vars), dtype=float).tocsr()
    constraints = LinearConstraint(A=A, lb=np.asarray(lower), ub=np.asarray(upper))
    bounds = Bounds(lb=lb, ub=ub)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": float(time_limit_s), "disp": False},
    )

    status_map = {
        0: "optimal",
        1: "iteration_or_time_limit",
        2: "infeasible",
        3: "unbounded",
        4: "numerical_issue",
    }
    status_text = status_map.get(int(result.status), f"status_{result.status}")

    if not result.success or result.x is None:
        return False, status_text, None, None, None

    x_values = result.x[x_idx.reshape(-1)].reshape(num_jobs, num_splits)
    selected_splits = np.argmax(x_values, axis=1).astype(int)

    start_times = np.column_stack(
        (
            result.x[local_idx],
            result.x[comm_idx],
            result.x[remote_idx],
        )
    )

    return True, status_text, float(result.x[cmax_idx]), selected_splits, start_times


def to_ms(seconds: float) -> float:
    return float(seconds) / 0.001


def pretty_time(seconds: float) -> str:
    return f"{seconds:.6f} s ({to_ms(seconds):.3f} ms)"


def print_job_details(label: str, splits: np.ndarray, stage_times: np.ndarray) -> None:
    print(label)
    print("job  split  local_s      comm_s       remote_s     total_s")
    for j in range(stage_times.shape[0]):
        l_t, c_t, r_t = stage_times[j]
        total = l_t + c_t + r_t
        print(f"{j:>3d}  {splits[j]:>5d}  {l_t:>10.6f}  {c_t:>11.6f}  {r_t:>11.6f}  {total:>10.6f}")


def main() -> None:
    args = parse_args()

    remote_layers, local_layers, comms, r, R = build_job_samples(
        num_jobs=args.num_jobs,
        stage=args.stage,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        log_uniform=args.log_uniform,
        comms_uniform=args.comms_uniform,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        seed=args.seed,
    )

    stage_options = build_stage_options(local_layers, remote_layers, comms)
    dopart_splits = pick_dopart_splits(local_layers, remote_layers, comms, r, R)
    dopart_stage_times = stage_times_from_splits(stage_options, dopart_splits)

    dopart_fifo = simulate_pipeline(dopart_stage_times)

    # Offline optimal scheduler for fixed DOPart splits (isolates scheduling effect).
    dopart_fixed_options = dopart_stage_times[:, None, :]
    dopart_sched_ok, dopart_sched_status, dopart_sched_makespan, _, dopart_starts = solve_optimal_schedule(
        dopart_fixed_options, time_limit_s=args.time_limit
    )

    # Full offline optimal: choose split + schedule jointly.
    opt_ok, opt_status, opt_makespan, opt_splits, opt_starts = solve_optimal_schedule(
        stage_options, time_limit_s=args.time_limit
    )

    print("=== DNN Pipeline with DOPart ===")
    print(f"Jobs: {args.num_jobs}")
    print(f"Layers per job: {remote_layers.size}")
    print(f"Seed: {args.seed}")
    print(f"DOPart ratio bounds used: r={r:.6f}, R={R:.6f}")
    print()

    print("Constraint model:")
    print("1) No overlap on local processing resource")
    print("2) No overlap on communication resource")
    print("3) No overlap on remote processing resource")
    print()

    print(f"DOPart FIFO makespan: {pretty_time(dopart_fifo.makespan)}")
    if dopart_sched_ok and dopart_sched_makespan is not None:
        dopart_order = np.argsort(dopart_starts[:, 0]).astype(int)
        print(f"DOPart best offline schedule makespan: {pretty_time(dopart_sched_makespan)}")
        print(f"DOPart best offline order: {dopart_order.tolist()}")
    else:
        print(f"DOPart best offline schedule solve failed: {dopart_sched_status}")

    if opt_ok and opt_makespan is not None and opt_splits is not None:
        opt_order = np.argsort(opt_starts[:, 0]).astype(int)
        print(f"Offline optimal makespan: {pretty_time(opt_makespan)}")
        print(f"Offline optimal order: {opt_order.tolist()}")
        print(f"Offline optimal splits: {opt_splits.tolist()}")
        print()
        print(f"DOPart FIFO / Offline OPT: {dopart_fifo.makespan / opt_makespan:.6f}x")
        if dopart_sched_ok and dopart_sched_makespan is not None:
            print(f"DOPart best offline schedule / Offline OPT: {dopart_sched_makespan / opt_makespan:.6f}x")
    else:
        print(f"Offline optimal solve failed: {opt_status}")

    if args.print_jobs:
        print()
        print_job_details("Per-job DOPart choices:", dopart_splits, dopart_stage_times)
        if opt_ok and opt_splits is not None:
            opt_stage_times = stage_times_from_splits(stage_options, opt_splits)
            print()
            print_job_details("Per-job offline optimal choices:", opt_splits, opt_stage_times)


if __name__ == "__main__":
    main()
