Comparative Study of DOPart and DOPart-AR
=========================================

Scope
-----
This document consolidates all requested analyses comparing DOPart vs DOPart-AR, including:
1) Why DOPart-AR was better at alpha values 0.0 and 0.5 in the specified mobilenet_v2 run.
2) Why DOPart-AR is worse at many other alpha values.
3) A parameter search over alpha-min/alpha-max/mid-scale for mobilenet_v2.
4) A full sweep across all profile models for:
   - alpha_min = -1
   - alpha_max in {2, 3}
   - mid_scale in {0.7, 1.0, 1.3}

Unless noted otherwise, runs used:
- log_uniform = True
- alpha_fixed = False
- comms_uniform = False
- random_min = True
- seed = 42
- NUM_SAMPLES = 7000


1) Why DOPart-AR beat DOPart at alpha = 0.0 and 0.5 (mobilenet_v2, alpha_min=-1, alpha_max=3, mid_scale=0.8)
---------------------------------------------------------------------------------------------------------------
Command context:
python .\project\plot.py --stages 0 --no-comms-uniform --log-uniform --alpha-min -1 --alpha-max 3 --no-alpha-fixed --random-min --ci 95 --seed 42 --mid-scale 0.8 --profile-model mobilenet_v2

Observed means:
- alpha = 0.0: DOPart = 79.2111, DOPart-AR = 76.9548, diff (AR-DOPart) = -2.2563
- alpha = 0.5: DOPart = 83.5396, DOPart-AR = 82.2281, diff (AR-DOPart) = -1.3115

Mechanism:
- DOPart uses a fixed global (a, b) thresholding logic across stages.
- DOPart-AR recomputes stage-adaptive bounds from prefix/suffix state and uses adaptive randomized thresholding.
- In these two alpha points, DOPart-AR chooses later offload points on average:
  - alpha 0.0: mean offload index ~6.09 vs DOPart ~3.57
  - alpha 0.5: mean offload index ~5.97 vs DOPart ~3.54
- Later offloading increased local compute, but reduced communication and remaining remote suffix enough to lower total makespan.

Component breakdown (AR minus DOPart):
- alpha 0.0: local +12.08, comm -10.73, remote -3.60, total -2.26
- alpha 0.5: local +13.48, comm -11.26, remote -3.53, total -1.31


2) Why DOPart-AR is worse at many other alpha values (same mobilenet_v2 setup)
--------------------------------------------------------------------------------
Across alpha grid [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], mean diff (AR-DOPart):
- -1.0: +2.3074
- -0.5: +0.3144
-  0.0: -2.2563
-  0.5: -1.3115
-  1.0: -0.0979 (near tie; AR slightly better)
-  1.5: +1.9370
-  2.0: +4.5686
-  2.5: +7.7033
-  3.0: +10.8438

Explanation:
- DOPart-AR generally offloads later than DOPart.
- This reliably reduces communication + remote suffix, but increases local compute.
- At moderate alpha (~0 to 1), comm/remote savings can outweigh local penalty.
- At high alpha (>=1.5), local penalty grows and dominates, so DOPart-AR becomes worse.
- At very low alpha (-1, -0.5), savings are not enough to offset the local increase in this setup.


3) Search over alpha-min/alpha-max/mid-scale for mobilenet_v2
--------------------------------------------------------------
Search grid:
- alpha_min in {-1.0, -0.5, 0.0, 0.5, 1.0}
- alpha_max in {2.5, 3.0, 3.5, 4.0}
- mid_scale in {0.6, 0.8, 1.0, 1.2}
- period = 0.5

Summary:
- scanned combos: 80
- combos with at least one alpha where AR better: 28
- combos where AR better for all alphas: 0
- combos where AR better on average over all alphas: 0

Validated notable combos where AR had improving alpha points:
- (-1.0, 4.0, 0.8): better at alpha {-1.0, -0.5, 0.0, 0.5, 1.0}, best diff -3.2870
- (-1.0, 3.5, 0.8): better at {0.0, 0.5, 1.0}, best diff -2.4589
- (-1.0, 3.0, 0.8): better at {0.0, 0.5, 1.0}, best diff -2.2563
- (-1.0, 4.0, 1.2): better at {-1.0, -0.5, 0.0, 0.5}, best diff -2.3183
- (-0.5, 4.0, 0.8): better at {-0.5, 0.0, 0.5, 1.0}, best diff -3.0551
- (-0.5, 4.0, 1.0): better at {-0.5, 0.0, 0.5}, best diff -2.3953
- (0.0, 4.0, 1.0): better at {0.0, 0.5}, best diff -2.0399
- (-1.0, 3.5, 1.2): better at {0.0, 0.5}, best diff -1.4533

Format above is (alpha_min, alpha_max, mid_scale).


4) Full sweep for all profile models with alpha_min=-1, alpha_max in {2,3}, mid_scale in {0.7,1.0,1.3}
---------------------------------------------------------------------------------------------------------
Total combinations:
- 11 profile models * 2 alpha_max values * 3 mid_scale values = 66 combos

Summary:
- combos with any alpha where AR better: 26
- total improving points (model, alpha_max, mid_scale, alpha): 66

Scenarios where AR is better (listed as improving alpha values):

lenet5
- alpha_max=2.0, mid_scale=1.0 -> alpha {-1.0}
- alpha_max=3.0, mid_scale=0.7 -> alpha {-0.5, 0.0, 0.5}
- alpha_max=3.0, mid_scale=1.0 -> alpha {-1.0, -0.5}
- alpha_max=3.0, mid_scale=1.3 -> alpha {-0.5, 0.0, 0.5}

alexnet
- alpha_max=2.0, mid_scale=0.7 -> alpha {0.0}

mobilenet_v3_small
- alpha_max=2.0, mid_scale=0.7 -> alpha {0.0, 0.5, 1.0}
- alpha_max=2.0, mid_scale=1.0 -> alpha {0.0, 0.5, 1.0}
- alpha_max=2.0, mid_scale=1.3 -> alpha {0.0, 0.5}
- alpha_max=3.0, mid_scale=0.7 -> alpha {-0.5}
- alpha_max=3.0, mid_scale=1.0 -> alpha {0.0, 0.5}
- alpha_max=3.0, mid_scale=1.3 -> alpha {0.0}

efficientnet_b0
- alpha_max=2.0, mid_scale=1.0 -> alpha {0.0}
- alpha_max=2.0, mid_scale=1.3 -> alpha {0.0}
- alpha_max=3.0, mid_scale=0.7 -> alpha {-0.5, 0.0, 0.5}
- alpha_max=3.0, mid_scale=1.0 -> alpha {0.0}
- alpha_max=3.0, mid_scale=1.3 -> alpha {0.0}

mobilenet_v2
- alpha_max=2.0, mid_scale=1.0 -> alpha {0.0}
- alpha_max=2.0, mid_scale=1.3 -> alpha {0.0}
- alpha_max=3.0, mid_scale=0.7 -> alpha {-0.5, 0.0, 0.5, 1.0}
- alpha_max=3.0, mid_scale=1.0 -> alpha {-0.5, 0.0}
- alpha_max=3.0, mid_scale=1.3 -> alpha {0.0, 0.5}

resnet152
- alpha_max=3.0, mid_scale=0.7 -> alpha {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
- alpha_max=3.0, mid_scale=1.3 -> alpha {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}

resnet200
- alpha_max=2.0, mid_scale=1.3 -> alpha {-0.5}
- alpha_max=3.0, mid_scale=0.7 -> alpha {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
- alpha_max=3.0, mid_scale=1.3 -> alpha {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}

Models with no improving point in this grid:
- squeezenet1_1
- shufflenet_v2_x1_0
- resnet18
- resnet34


5) Notes
--------
- Results are Monte Carlo estimates; seed fixed at 42 for repeatability.
- commsRange itself is stochastic; sweeps were run with controlled seeding.
- All comparisons above use mean makespan over NUM_SAMPLES.
