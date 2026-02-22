# DNN Stage Profiling

## Purpose
`profile_resnet_stage_stats.py` profiles per-stage delay and stage-input size for:
- `lenet5`
- `alexnet`
- `squeezenet1_1`
- `mobilenet_v3_small`
- `efficientnet_b0`
- `shufflenet_v2_x1_0`
- `mobilenet_v2`
- `resnet18`
- `resnet34`
- `resnet152`
- `resnet200`

Stage means model-specific major block:
- ResNets: residual blocks (`layer1.0` ... `layer4.N`)
- MobileNets: `features.*` blocks
- EfficientNet-B0: stem + each MBConv + head
- ShuffleNetV2: `conv1` + all stage units + `conv5`
- SqueezeNet1.1: stem conv + Fire modules + classifier conv
- AlexNet: conv layers + linear layers
- LeNet-5: `conv1`, `conv2`, `fc1`, `fc2`, `fc3`

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
- `lenet5`: custom compact LeNet-style module (5-stage profile)
- `alexnet`: `torchvision.models.alexnet(weights=None)`
- `squeezenet1_1`: `torchvision.models.squeezenet1_1(weights=None)`
- `mobilenet_v3_small`: `torchvision.models.mobilenet_v3_small(weights=None)`
- `efficientnet_b0`: `torchvision.models.efficientnet_b0(weights=None)`
- `shufflenet_v2_x1_0`: `torchvision.models.shufflenet_v2_x1_0(weights=None)`
- `mobilenet_v2`: `torchvision.models.mobilenet_v2(weights=None)`
- `resnet18`: `torchvision.models.resnet18(weights=None)`
- `resnet34`: `torchvision.models.resnet34(weights=None)`
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
.\.venv\Scripts\python "DNNs\profile_resnet_stage_stats.py" --models mobilenet_v2 --runs 100 --warmup 10 --out-dir "DNNs"
```

Run multiple small models:

```powershell
.\.venv\Scripts\python "DNNs\profile_resnet_stage_stats.py" --models alexnet squeezenet1_1 mobilenet_v3_small efficientnet_b0 shufflenet_v2_x1_0 mobilenet_v2 resnet18 resnet34 lenet5 --runs 100 --warmup 10 --out-dir "DNNs"
```

## Use With `project/plot.py`
After generating profile files in `DNNs/`, run:

```powershell
python .\project\plot.py --profile-model mobilenet_v2 --stages 0 --no-comms-uniform --log-uniform --alpha-min -1 --alpha-max 3 --no-alpha-fixed --lower-bound 0.05 --upper-bound 2.5 --random-min --ci 95 --seed 42
```

Generate only stage profile plots for a profiled model:

```powershell
python .\project\plot.py --profile-model resnet152 --stage-plots-only --no-show
```

Notes:
- `--profile-model` loads `DNNs/<model>_stage_avg_times_ms.txt` and `DNNs/<model>_stage_input_sizes_bytes.txt`.
- Legacy behavior is preserved: `--stages 152` maps to `resnet152`, and `--stages 200` maps to `resnet200`.
- If both are provided, `--profile-model` is used for stage profile loading.

## CLI Arguments
- `--models`: one or more of:
  - `lenet5 alexnet squeezenet1_1 mobilenet_v3_small efficientnet_b0 shufflenet_v2_x1_0 mobilenet_v2 resnet18 resnet34 resnet152 resnet200`
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
