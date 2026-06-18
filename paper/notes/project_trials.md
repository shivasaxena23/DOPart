# Project Trials Log

Last updated: 2026-06-03

This is a living log of what we have tried so far in the DOPart project. Keep
new entries short but specific: include the setting, script/command if relevant,
what happened, and what we concluded.

## Current Claim Being Investigated

DOPart-R should not be presented as universally better than DOPart. The useful
claim is more specific:

> DOPart-R is useful in volatile hybrid offloading regimes where deterministic
> DOPart commits too early, while OPT lies at early interior partition points.

The strongest current empirical setting is:

```text
Profile: ResNet34
alpha_min = 0
alpha_max = 4
communication range factor = 3
beta concentration = 0.2
samples = 2500
```

Observed mean regret:

```text
DOPart         97.756
DOPart-R       48.688
DSR            82.229
Always Offload 97.756
Never Offload  265.023
OPT             0.000
```

## Algorithm Alignment Work

### Renamed Algorithms To Match Paper v32

Files touched:

```text
project/methods.py
project/plot.py
```

Changes:

- Renamed static randomized helper from `DOPartRAND` to `DSR`.
- Renamed adaptive randomized algorithm from `DOPartARAND` to `DOPart-R`.
- Kept backwards-compatible aliases:

```text
DOPartRAND = DSR
DOPartARAND = DOPartR
```

Conclusion:

The code names now better match the v32 paper terminology.

### Paper-Style Alpha Bounds

Added paper-style bounds:

```text
r' = min(r, 1)
R' = max(R, 1)
```

Applied to:

- `DSR`
- `DOPart-R`
- randomized threshold parameters in plotting code

Conclusion:

The randomized algorithms are now closer to the paper’s theoretical setup.

## Alpha Sampling Questions

### Question

What changes when running:

```powershell
python .\project\plot.py --stages 0 --no-comms-uniform --log-uniform --alpha-min -1 --alpha-max 3 --no-alpha-fixed --random-min --ci 95 --seed 42 --stage-plots --profile-model resnet34
```

Conclusion:

With `--no-alpha-fixed`, the sweep changes `alpha_min`, while `alpha_max` stays
fixed. To make `alpha_max` change, use the default `--alpha-fixed` behavior or
explicitly pass `--alpha-fixed`.

### Question

Why can the offload index stay constant even when `alpha_max` increases?

Conclusion:

If the communication delay or model profile strongly favors one side of the
partition, increasing `alpha_max` may change the cost values without changing
the threshold crossing point. The offload layer can remain fixed even when the
makespan changes.

## OPT Distribution Observations

We observed that the distribution of OPT offload points can be skewed:

- toward early layers when communication makes later offload unattractive
- toward the final layer when local computation is often favorable
- across interior layers only in hybrid regimes

Conclusion:

DOPart-R is more interesting when OPT is not concentrated at a single endpoint.
If OPT is almost always first-stage or final-stage, a trivial baseline can look
very strong.

## Why DSR / DOPart-R Sometimes Look Bad

Question:

Why does DSR or DOPart-R do worse than DOPart when DOPart is close to OPT?

Conclusion:

DOPart is deterministic and very strong when the best threshold decision is
stable. Randomization helps only when deterministic thresholding repeatedly
makes the same wrong kind of commitment. If the problem distribution is stable,
randomization adds variance without enough benefit.

## Synthetic DOPart-R Demonstrations

Script:

```text
project/demo_dopartr_beats_dopart.py
```

Purpose:

Create a controlled demonstration where DOPart-R can beat DOPart.

What we learned:

- It is easy to make synthetic settings where DOPart-R beats DOPart.
- Some early versions were too synthetic because OPT was always first-stage or
  always near an endpoint.
- Once `Always Offload` and `Never Offload` were added, some apparent DOPart-R
  wins became less convincing because a trivial baseline performed even better.

Conclusion:

Synthetic examples are useful for intuition, but the paper case needs a
realistic profile and nontrivial baselines.

## Communication Range Sweep

Script:

```text
project/plot_comms_range_sweep.py
experiments/comms_range_sweep.py
```

Purpose:

Study what happens to DOPart, DOPart-R, DSR, OPT, Always Offload, and Never
Offload as communication variability changes.

Communication delay model:

```text
communication_delay_i = input_size_i / bandwidth_i
```

Bandwidth is sampled from a beta distribution over:

```text
[base_bandwidth / range_factor, base_bandwidth * range_factor]
```

The mean bandwidth is kept fixed at `base_bandwidth`.

Conclusion:

Increasing communication variability alone does not guarantee that DOPart-R
beats DOPart. The alpha range and OPT distribution matter.

## Beta Distribution Sweep

Script:

```text
project/plot_beta_distributions.py
experiments/beta_distributions.py
```

Purpose:

Visualize the beta bandwidth distributions used in the examples.

Key idea:

- Low beta concentration, such as `0.2`, creates a U-shaped distribution.
- Higher beta concentration concentrates bandwidth near the fixed mean.

Conclusion:

Low beta concentration is the most promising for DOPart-R because the network
conditions are volatile while the mean remains unchanged.

## Searching For DOPart-R Winning Regimes

### First Search

Setting:

```text
profile = resnet34
alpha_min = -1
alpha_max = 3
range_factors = [2, 5, 10, 20, 50, 100]
beta_concentrations = [0.5, 1, 2, 5, 10, 20, 50, 100, 200]
```

Result:

No tested beta distribution made DOPart-R beat DOPart.

Conclusion:

The original alpha range is not a good empirical case for DOPart-R.

### Wider Alpha Search

Promising settings appeared when alpha ranges were widened or shifted.

Examples:

```text
alpha_min = -4, alpha_max = 2
range_factor = 2
beta_concentration = 0.2
```

This made DOPart-R beat DOPart, but the setting was often close to
`Never Offload`, weakening the case once trivial baselines were included.

Conclusion:

DOPart-R beating DOPart is not enough. The setting must also beat the trivial
baselines.

### Cross-Profile Search

We tested candidate regimes across several model profiles.

The cleanest setting found:

```text
profile = resnet34
alpha_min = 0
alpha_max = 4
range_factor = 3
beta_concentration = 0.2
```

This setting also worked strongly for other profiles such as AlexNet,
EfficientNet-B0, MobileNet variants, ResNet18, ResNet152, and ResNet200.

Conclusion:

The best DOPart-R case is a volatile hybrid-link setting where:

- local compute is equal to slower than remote
- bandwidth is volatile around a fixed mean
- DOPart tends to offload too early
- OPT usually lies near early interior layers
- DOPart-R beats DOPart and trivial baselines

## DOPart-R Case Study Figures

Script:

```text
project/plot_dopartr_case_studies.py
experiments/dopartr_case_studies.py
```

Generated figures:

```text
outputs/plots/archive/dopartr_case_beta_sweep.pdf
outputs/plots/archive/dopartr_case_scenario_regret.pdf
outputs/plots/archive/dopartr_case_opt_offload_hist.pdf
outputs/plots/archive/dopartr_case_regret_cdf.pdf
```

Scenarios:

### Stable Communication Control

```text
alpha_min = -1
alpha_max = 3
range_factor = 2
beta_concentration = 80
```

Observation:

DOPart-R is worse than DOPart.

Conclusion:

This is a boundary/control case showing that randomization is not always
beneficial.

### Volatile Hybrid Link

```text
alpha_min = 0
alpha_max = 4
range_factor = 3
beta_concentration = 0.2
```

Observation:

DOPart-R beats DOPart, DSR, Always Offload, and Never Offload.

Conclusion:

This is the current strongest case for DOPart-R.

### Wider-Range Boundary

```text
alpha_min = 0
alpha_max = 4
range_factor = 10
beta_concentration = 1
```

Observation:

DOPart recovers and DOPart-R is slightly worse.

Conclusion:

The DOPart-R claim should be tied to the volatile hybrid-link regime, not stated
as universal dominance.

## Baselines Added

Added to communication sweep and case-study experiments:

```text
Always Offload
Never Offload
```

Definitions:

```text
Always Offload cost = comms[0] + sum(remote)
Never Offload cost  = sum(local)
```

Conclusion:

These baselines are essential. They prevent us from claiming DOPart-R is useful
in cases where a trivial endpoint policy is already better.

## Project Organization Work

Added clean package:

```text
dopart/
  algorithms.py
  profiles.py
  sampling.py
  evaluation.py
  plotting.py
```

Added experiment wrappers:

```text
experiments/alpha_sweep.py
experiments/comms_range_sweep.py
experiments/beta_distributions.py
experiments/dopartr_case_studies.py
experiments/synthetic_dopartr_demo.py
```

Moved loose content into:

```text
data/
notebooks/
outputs/
paper/
theory/
```

Loader update:

`project/data_generation.py` now finds `resnet34_compute_values_224_t4.npy`
under `data/raw/`, with a fallback to the old root location.

Smoke tests run:

```powershell
python -c "from project.data_generation import system_values; r,s=system_values(0); print(len(r), len(s))"
python .\experiments\comms_range_sweep.py --samples 2 --range-min 1 --range-max 1 --no-show
```

Conclusion:

The repo is now closer to:

- reusable implementation in `dopart/`
- reproducible experiment scripts in `experiments/`
- notes and paper materials in `paper/`
- generated outputs in `outputs/`

## Dispersion / Data-Driven DOPart Search

Date: 2026-06-17

Folder:

```text
Dispersion/
```

Script:

```text
Dispersion/find_dopart_gaps.py
```

Purpose:

Find scenario distributions where deterministic DOPart is significantly worse
than OPT, and therefore where data-driven algorithm design may have room to
learn a better DOPart variant.

First learnable knob:

```text
DOPart(theta):
  offload at i if comm_i <= theta * DOPartThreshold_i
```

`theta = 1` recovers the original deterministic DOPart.

Medium scan command:

```powershell
python .\Dispersion\find_dopart_gaps.py --profiles resnet34 --alpha-ranges -1:3 0:3 0:4 1:4 --range-factors 2 3 5 10 --beta-concentrations 0.2 0.5 1 2 5 20 80 --samples 200 --seeds 2 --theta-count 25 --top-k 15
```

Output:

```text
Dispersion/results/dopart_gap_scenarios.csv
```

Top scenario found:

```text
profile = resnet34
alpha_min = 0
alpha_max = 4
range_factor = 3
beta_concentration = 0.2
DOPart regret = 96.584
best theta = 0.400
learnable improvement = 91.745
learnable improvement as pct of DOPart gap = 94.99%
OPT interior fraction = 74.25%
```

Conclusion:

The same volatile hybrid-link setting remains the strongest candidate, but now
we have a dispersion-style diagnostic showing that a simple learned threshold
multiplier could close much of DOPart's gap to OPT. This is a promising place to
start implementing ERM / online learning / private optimization over the DOPart
parameter.

## Open Questions

1. Should final paper plots use only ResNet34 or multiple profiles?
2. Should DOPart-R case-study figures report regret, competitive ratio, or both?
3. Should we include the beta distribution PDF beside the DOPart-R case figures?
4. Should `DNNs/` be migrated fully into `data/profiles/`?
5. Should old `project/` scripts be converted into true wrappers and the logic
   moved fully into `dopart/`?

## How To Add Future Entries

Use this format:

```text
## Short Experiment Name

Date:
Script:
Command:
Setting:
Observation:
Conclusion:
Files/plots:
```
