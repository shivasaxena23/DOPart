from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DNN_DIR = REPO_ROOT / "DNNs"
PROFILE_MODELS = (
    "lenet5",
    "alexnet",
    "squeezenet1_1",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "shufflenet_v2_x1_0",
    "mobilenet_v2",
    "resnet18",
    "resnet34",
    "resnet152",
    "resnet200",
)

LEGACY_STAGE_PROFILE_MAP = {
    152: "resnet152",
    200: "resnet200",
}


def _load_stage_profile(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    times_path = DNN_DIR / f"{model_name}_stage_avg_times_ms.txt"
    sizes_path = DNN_DIR / f"{model_name}_stage_input_sizes_bytes.txt"

    if not times_path.exists() or not sizes_path.exists():
        raise FileNotFoundError(
            f"Missing profile files for {model_name}. Expected: {times_path} and {sizes_path}"
        )

    compute_values_remote = np.atleast_1d(np.loadtxt(times_path, dtype=float))
    input_data_real = np.atleast_1d(np.loadtxt(sizes_path, dtype=float))

    if compute_values_remote.size != input_data_real.size:
        raise ValueError(
            "Profile length mismatch: "
            f"{times_path.name} has {compute_values_remote.size}, "
            f"but {sizes_path.name} has {input_data_real.size}."
        )
    return compute_values_remote, input_data_real


def system_values(i, profile_model: str | None = None):
    selected_profile = profile_model
    if selected_profile is not None and selected_profile not in PROFILE_MODELS:
        allowed = ", ".join(PROFILE_MODELS)
        raise ValueError(f"Unsupported profile model '{selected_profile}'. Choose one of: {allowed}")
    if selected_profile is None:
        selected_profile = LEGACY_STAGE_PROFILE_MAP.get(i)

    if selected_profile is not None:
        compute_values_remote, input_data_real = _load_stage_profile(selected_profile)
    elif i == 0:
        with (REPO_ROOT / "resnet34_compute_values_224_t4.npy").open("rb") as f:
            model_compute_values = np.load(f, allow_pickle=True)
        model_compute_values_remote = model_compute_values[1000:10000, :]  # 1000 or 10000
        compute_values_remote = np.mean(model_compute_values_remote, axis=0)
        input_data_real = np.array(
            [
                224 * 224 * 3,
                64 * 112 * 112,
                64 * 56 * 56,
                64 * 56 * 56,
                64 * 56 * 56,
                64 * 56 * 56,
                64 * 56 * 56,
                64 * 56 * 56,
                64 * 56 * 56,
                128 * 28 * 28,
                128 * 28 * 28,
                128 * 28 * 28,
                128 * 28 * 28,
                128 * 28 * 28,
                128 * 28 * 28,
                128 * 28 * 28,
                256 * 14 * 14,
                256 * 14 * 14,
                256 * 14 * 14,
                256 * 14 * 14,
                256 * 14 * 14,
                256 * 14 * 14,
                256 * 14 * 14,
                256 * 14 * 14,
                256 * 14 * 14,
                256 * 14 * 14,
                256 * 14 * 14,
                512 * 7 * 7,
                512 * 7 * 7,
                512 * 7 * 7,
                512 * 7 * 7,
                512 * 7 * 7,
                512 * 1 * 1,
            ],
            dtype=float,
        )
    elif i > 0:
        compute_values_remote = np.concatenate([np.zeros(i - 1), [10]])
        input_data_real = np.concatenate([[10] * i]).astype(float)
    else:
        raise ValueError(
            "stages must be 0, 152, 200, or a positive integer. "
            "You can also set profile_model to one of: "
            + ", ".join(PROFILE_MODELS)
            + "."
        )

    print(len(compute_values_remote), len(input_data_real))
    return compute_values_remote, input_data_real
