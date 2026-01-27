import torch
import torch.nn as nn
import torchvision.models as models
import time
import math
from collections import defaultdict, OrderedDict
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
use_cuda = torch.cuda.is_available()

model = models.resnet152(weights=None).to(device)
model.eval()

input_tensor = torch.randn(1, 3, 224, 224).to(device)

num_runs = 100  # Number of runs for averaging

# Data structures
layer_stats = defaultdict(lambda: {
    'name': '',
    'input_size': None,
    'total_runtime': 0.0,
    'count': 0
})
start_times = {}
start_events = {}

# Hooks
def register_named_hook(module, name):
    def pre_hook(mod, inp):
        if use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()                   # record event in current stream
            start_events[mod] = ev
        else:
            start_times[mod] = time.time()
        # store name and input size for this module
        layer_stats[mod]['name'] = name
        layer_stats[mod]['input_size'] = list(inp[0].shape)

    def post_hook(mod, inp, out):
        if use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            end_event = ev
            # wait for the end event to complete
            end_event.synchronize()
            runtime_ms = start_events[mod].elapsed_time(end_event)
            layer_stats[mod]['total_runtime'] += runtime_ms
        else:
            end_time = time.time()
            runtime_ms = (end_time - start_times[mod]) * 1000
            layer_stats[mod]['total_runtime'] += runtime_ms

        layer_stats[mod]['count'] += 1

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)

# Initial layers
initial_layer_names = {
    'conv1': model.conv1,
    'bn1': model.bn1,
    'relu': model.relu,
    'maxpool': model.maxpool
}
for name, layer in initial_layer_names.items():
    register_named_hook(layer, name)

# Residual blocks
for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    layer = getattr(model, layer_name)
    for idx, block in enumerate(layer):
        block_name = f"{layer_name}_block{idx+1}"
        register_named_hook(block, block_name)

# Forward pass multiple times
with torch.no_grad():
    for _ in range(num_runs):
        model(input_tensor)

# === Write runtimes ===
with open("resnet152_avg_runtime.txt", "w") as f:
    for info in layer_stats.values():
        avg_runtime = info['total_runtime'] / info['count']
        f.write(f"{avg_runtime:.4f}\n")

# === Write input sizes in bytes ===
with open("resnet152_input_size_bytes.txt", "w") as f:
    for info in layer_stats.values():
        shape = info['input_size']
        num_elements = np.prod(shape)
        size_bytes = num_elements * 4  # float32
        f.write(f"{size_bytes}\n")

print("Runtime and input size profiling complete.")
