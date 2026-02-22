import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torchvision.models as models
from torch import nn
from torchvision.models.resnet import Bottleneck, ResNet


@dataclass
class StageStat:
    input_bytes: int = 0
    total_ms: float = 0.0
    count: int = 0
    cpu_start: float = 0.0
    cuda_start_event: torch.cuda.Event | None = None


SUPPORTED_MODELS = (
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


class LeNet5(nn.Module):
    """Compact LeNet-style network with five profiled stages."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adapt_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.pool(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = self.adapt_pool(x)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile per-stage delay and stage input sizes for multiple DNN backbones. "
            "Each model uses block-level stages (e.g., residual blocks, inverted residuals)."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet152", "resnet200"],
        choices=SUPPORTED_MODELS,
        help="Models to profile.",
    )
    parser.add_argument("--runs", type=int, default=100, help="Measured forward passes.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup forward passes.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. auto uses CUDA if available, otherwise CPU.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory for output files. Default is this script's directory.",
    )
    return parser.parse_args()


def build_model(name: str) -> torch.nn.Module:
    if name == "lenet5":
        return LeNet5(num_classes=1000)
    if name == "alexnet":
        return models.alexnet(weights=None)
    if name == "squeezenet1_1":
        return models.squeezenet1_1(weights=None)
    if name == "mobilenet_v3_small":
        return models.mobilenet_v3_small(weights=None)
    if name == "efficientnet_b0":
        return models.efficientnet_b0(weights=None)
    if name == "shufflenet_v2_x1_0":
        return models.shufflenet_v2_x1_0(weights=None)
    if name == "mobilenet_v2":
        return models.mobilenet_v2(weights=None)
    if name == "resnet18":
        return models.resnet18(weights=None)
    if name == "resnet34":
        return models.resnet34(weights=None)
    if name == "resnet152":
        return models.resnet152(weights=None)
    if name == "resnet200":
        # Canonical ResNet-200 bottleneck layout.
        return ResNet(Bottleneck, [3, 24, 36, 3], num_classes=1000)
    raise ValueError(f"Unsupported model: {name}")


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_resnet_stage_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    stage_modules: list[tuple[str, torch.nn.Module]] = []
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, layer_name)
        for block_idx, block in enumerate(layer):
            stage_modules.append((f"{layer_name}.{block_idx}", block))
    return stage_modules


def collect_lenet5_stage_modules(model: LeNet5) -> list[tuple[str, torch.nn.Module]]:
    return [
        ("conv1", model.conv1),
        ("conv2", model.conv2),
        ("fc1", model.fc1),
        ("fc2", model.fc2),
        ("fc3", model.fc3),
    ]


def collect_alexnet_stage_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [
        ("features.0", model.features[0]),
        ("features.3", model.features[3]),
        ("features.6", model.features[6]),
        ("features.8", model.features[8]),
        ("features.10", model.features[10]),
        ("classifier.1", model.classifier[1]),
        ("classifier.4", model.classifier[4]),
        ("classifier.6", model.classifier[6]),
    ]


def collect_squeezenet_stage_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    fire_ids = (3, 4, 6, 7, 9, 10, 11, 12)
    stage_modules = [("features.0", model.features[0])]
    stage_modules.extend((f"features.{i}", model.features[i]) for i in fire_ids)
    stage_modules.append(("classifier.1", model.classifier[1]))
    return stage_modules


def collect_mobilenet_v3_small_stage_modules(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    return [(f"features.{i}", model.features[i]) for i in range(len(model.features))]


def collect_efficientnet_b0_stage_modules(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    stage_modules: list[tuple[str, torch.nn.Module]] = [("features.0", model.features[0])]
    for group_idx in range(1, 8):
        block_group = model.features[group_idx]
        for block_idx, block in enumerate(block_group):
            stage_modules.append((f"features.{group_idx}.{block_idx}", block))
    stage_modules.append(("features.8", model.features[8]))
    return stage_modules


def collect_shufflenet_stage_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    stage_modules: list[tuple[str, torch.nn.Module]] = [("conv1", model.conv1)]
    for stage_name in ("stage2", "stage3", "stage4"):
        stage = getattr(model, stage_name)
        for block_idx, block in enumerate(stage):
            stage_modules.append((f"{stage_name}.{block_idx}", block))
    stage_modules.append(("conv5", model.conv5))
    return stage_modules


def collect_mobilenet_v2_stage_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [(f"features.{i}", model.features[i]) for i in range(len(model.features))]


def collect_stage_modules(model_name: str, model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    if model_name in {"resnet18", "resnet34", "resnet152", "resnet200"}:
        return collect_resnet_stage_modules(model)
    if model_name == "lenet5":
        if not isinstance(model, LeNet5):
            raise TypeError(f"Expected LeNet5 instance, got {type(model).__name__}.")
        return collect_lenet5_stage_modules(model)
    if model_name == "alexnet":
        return collect_alexnet_stage_modules(model)
    if model_name == "squeezenet1_1":
        return collect_squeezenet_stage_modules(model)
    if model_name == "mobilenet_v3_small":
        return collect_mobilenet_v3_small_stage_modules(model)
    if model_name == "efficientnet_b0":
        return collect_efficientnet_b0_stage_modules(model)
    if model_name == "shufflenet_v2_x1_0":
        return collect_shufflenet_stage_modules(model)
    if model_name == "mobilenet_v2":
        return collect_mobilenet_v2_stage_modules(model)
    raise ValueError(f"Unsupported model for stage extraction: {model_name}")


def profile_model(
    model_name: str,
    model_builder: Callable[[str], torch.nn.Module],
    device: torch.device,
    runs: int,
    warmup: int,
    batch_size: int,
    height: int,
    width: int,
) -> tuple[list[str], list[float], list[int]]:
    if runs <= 0:
        raise ValueError("--runs must be positive.")
    if warmup < 0:
        raise ValueError("--warmup cannot be negative.")

    model = model_builder(model_name).to(device)
    model.eval()

    stage_modules = collect_stage_modules(model_name, model)
    stage_names = [name for name, _ in stage_modules]
    stats = {name: StageStat() for name in stage_names}

    use_cuda = device.type == "cuda"
    state = {"collect": False}

    def register_hooks(name: str, module: torch.nn.Module) -> None:
        def pre_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            input_tensor = inputs[0]
            if isinstance(input_tensor, torch.Tensor) and stats[name].input_bytes == 0:
                stats[name].input_bytes = int(input_tensor.numel() * input_tensor.element_size())

            if not state["collect"]:
                return

            if use_cuda:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                stats[name].cuda_start_event = ev
            else:
                stats[name].cpu_start = time.perf_counter()

        def post_hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            if not state["collect"]:
                return

            if use_cuda:
                start_ev = stats[name].cuda_start_event
                if start_ev is None:
                    return
                end_ev = torch.cuda.Event(enable_timing=True)
                end_ev.record()
                end_ev.synchronize()
                elapsed_ms = float(start_ev.elapsed_time(end_ev))
            else:
                elapsed_ms = (time.perf_counter() - stats[name].cpu_start) * 1000.0

            stats[name].total_ms += elapsed_ms
            stats[name].count += 1

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)

    for stage_name, stage_module in stage_modules:
        register_hooks(stage_name, stage_module)

    input_tensor = torch.randn(batch_size, 3, height, width, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)

        state["collect"] = True
        for _ in range(runs):
            model(input_tensor)

    if use_cuda:
        torch.cuda.synchronize()

    avg_times_ms: list[float] = []
    input_sizes_bytes: list[int] = []
    for name in stage_names:
        stat = stats[name]
        if stat.count == 0:
            raise RuntimeError(f"No measurements collected for stage {name}.")
        avg_times_ms.append(stat.total_ms / stat.count)
        input_sizes_bytes.append(stat.input_bytes)

    return stage_names, avg_times_ms, input_sizes_bytes


def write_profile_files(
    out_dir: Path,
    model_name: str,
    stage_names: list[str],
    avg_times_ms: list[float],
    input_sizes_bytes: list[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    names_path = out_dir / f"{model_name}_stage_names.txt"
    times_path = out_dir / f"{model_name}_stage_avg_times_ms.txt"
    sizes_path = out_dir / f"{model_name}_stage_input_sizes_bytes.txt"

    with names_path.open("w", encoding="utf-8") as f:
        for name in stage_names:
            f.write(f"{name}\n")

    with times_path.open("w", encoding="utf-8") as f:
        for value in avg_times_ms:
            f.write(f"{value:.6f}\n")

    with sizes_path.open("w", encoding="utf-8") as f:
        for value in input_sizes_bytes:
            f.write(f"{value}\n")

    print(f"[{model_name}] wrote:")
    print(f"  {names_path}")
    print(f"  {times_path}")
    print(f"  {sizes_path}")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    out_dir = Path(args.out_dir).resolve()

    print(f"Using device: {device}")
    print(f"Output directory: {out_dir}")

    for model_name in args.models:
        print(f"Profiling {model_name} ...")
        stage_names, avg_times_ms, input_sizes_bytes = profile_model(
            model_name=model_name,
            model_builder=build_model,
            device=device,
            runs=args.runs,
            warmup=args.warmup,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
        )
        write_profile_files(out_dir, model_name, stage_names, avg_times_ms, input_sizes_bytes)
        print(f"[{model_name}] stages profiled: {len(stage_names)}")


if __name__ == "__main__":
    main()
