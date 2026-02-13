# DNN Stage Profiling

## Purpose
`profile_resnet_stage_stats.py` profiles per-stage delay and stage-input size for:
- `resnet152`
- `resnet200`

Stage means residual block:
- `layer1.0` ... `layer4.N`

## What It Outputs
For each model, the script writes three files:
- `<model>_stage_names.txt`
- `<model>_stage_avg_times_ms.txt`
- `<model>_stage_input_sizes_bytes.txt`

Example:
- `resnet152_stage_names.txt`
- `resnet152_stage_avg_times_ms.txt`
- `resnet152_stage_input_sizes_bytes.txt`

## Model Definitions
- `resnet152`: `torchvision.models.resnet152(weights=None)`
- `resnet200`: `ResNet(Bottleneck, [3, 24, 36, 3], num_classes=1000)`

## Run
From repository root (recommended interpreter):

```powershell
.\.venv\Scripts\python "DNNs\profile_resnet_stage_stats.py" --runs 100 --warmup 10 --out-dir "DNNs"
```

CPU-only run:

```powershell
.\.venv\Scripts\python "DNNs\profile_resnet_stage_stats.py" --device cpu --runs 100 --warmup 10 --out-dir "DNNs"
```

Single model run:

```powershell
.\.venv\Scripts\python "DNNs\profile_resnet_stage_stats.py" --models resnet200 --runs 100 --warmup 10 --out-dir "DNNs"
```

## CLI Arguments
- `--models`: one or both of `resnet152 resnet200`
- `--runs`: measured forward passes
- `--warmup`: warmup forward passes
- `--batch-size`: input batch size (default `1`)
- `--height`, `--width`: input resolution (default `224x224`)
- `--device`: `auto`, `cpu`, or `cuda`
- `--out-dir`: output directory

## Notes
- Delay is measured in milliseconds per stage.
- Input size is written in bytes.
- `--device auto` uses CUDA if available, else CPU.
- Use `.venv\Scripts\python`; system Python may not include `torchvision`.
