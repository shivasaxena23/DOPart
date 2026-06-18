# Dispersion Experiments For DOPart

This folder is for data-driven algorithm design experiments around DOPart.

The first goal is to find distributions where deterministic DOPart has a large
performance gap relative to OPT. Those are the scenarios where learning may be
worthwhile.

## Current Search Tool

```powershell
python .\Dispersion\find_dopart_gaps.py --samples 500 --seeds 3 --top-k 20
```

The script scans scenario distributions over:

- DNN profile
- alpha range
- communication range factor
- beta bandwidth concentration

For each scenario it reports:

- DOPart mean regret over OPT
- DOPart mean competitive ratio
- endpoint baseline performance
- OPT offload distribution statistics
- best fixed threshold multiplier from a grid
- estimated learnable improvement from that multiplier

## Learnable Knob

The first learnable class is a one-parameter DOPart variant:

```text
DOPart(theta):
  offload at i if comm_i <= theta * DOPartThreshold_i
```

`theta = 1` is the original deterministic DOPart.

This is not yet the final dispersion-learning algorithm. It is a diagnostic
class for finding where learning has room to help. A scenario is especially
interesting when:

```text
DOPart regret over OPT is large
and
best theta has much lower regret than theta = 1
and
Always/Never Offload are not already enough
```

## Outputs

By default, generated files go under:

```text
Dispersion/results/
```

These outputs are ignored by Git except for `.gitkeep`.

## Next Step

Once promising scenarios are found, implement the actual dispersion/data-driven
training loop:

1. Generate training samples from the scenario.
2. Evaluate the piecewise DOPart(theta) loss over a theta grid.
3. Select theta by empirical risk minimization.
4. Evaluate on fresh test samples.
5. Repeat across seeds and privacy/noise settings if needed.

