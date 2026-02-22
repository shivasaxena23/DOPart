from datetime import datetime
import math
from pathlib import Path
import random
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data_generation import PROFILE_MODELS, system_values
from methods import TBP, DOPart, DOPartARAND, DOPartARANDR, DOPartRAND, DOPartRANDR, TBP_ratio, rand_threshold_params
import argparse

from findCommsRange import commsRange

parser = argparse.ArgumentParser()
parser.add_argument(
  "--stages",
  type=int,
  default=0,
  help="Synthetic stage count by default. Legacy profile aliases: 152->resnet152, 200->resnet200.",
)
parser.add_argument(
  "--profile-model",
  type=str,
  choices=PROFILE_MODELS,
  default=None,
  help="Load pre-profiled stage delay/input data from DNNs for this model name.",
)

# booleans
parser.add_argument("--comms-uniform", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--log-uniform", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--random-min", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
  "--offload-plot",
  action=argparse.BooleanOptionalAction,
  default=True,
  help="Generate an additional plot of offload layer indices per algorithm.",
)

# floats
parser.add_argument("--alpha-min", type=float, default=1.0)
parser.add_argument("--alpha-max", type=float, default=4.0)
parser.add_argument("--period", type=float, default=0.5)
parser.add_argument("--alpha-fixed", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--comms-range-factor", type=int, default=1)
parser.add_argument("--mid-scale", type=float, default=1)
parser.add_argument("--lower-bound", type=float, default=0.25)
parser.add_argument("--upper-bound", type=float, default=2.5)
parser.add_argument(
  "--stage-plots",
  action=argparse.BooleanOptionalAction,
  default=False,
  help="Generate stage-latency and stage-input-size plots from imported profile data.",
)
parser.add_argument(
  "--stage-plots-only",
  action=argparse.BooleanOptionalAction,
  default=False,
  help="Generate only stage profile plots and exit before algorithm simulation.",
)
parser.add_argument(
  "--stage-plot-out-dir",
  type=str,
  default=str(Path(__file__).resolve().parent / "plots"),
  help="Directory for stage profile plots.",
)
parser.add_argument(
  "--no-show",
  action="store_true",
  help="Disable interactive plot windows.",
)
parser.add_argument(
  "--ci",
  type=float,
  default=0.0,
  help="Confidence interval level for seaborn lineplot (e.g., 95). Use 0 to disable.",
)
parser.add_argument(
  "--seed",
  type=int,
  default=None,
  help="Random seed for reproducible sampling and CI bootstrapping.",
)

args = parser.parse_args()
NUM_SAMPLES = 7000
IGNORED_ALGS = {0, 2, 6, 8}

v = args.stages
comms_uniform = args.comms_uniform
lb = args.lower_bound
ub = args.upper_bound
comms_range_factor = args.comms_range_factor
mid_scale = args.mid_scale
log_uniform = args.log_uniform
alpha_fixed = args.alpha_fixed
alpha_min = args.alpha_min
alpha_max = args.alpha_max
period = args.period
random_min = args.random_min
offload_plot = args.offload_plot
ci_level = args.ci
seed = args.seed
profile_model = args.profile_model
print("Random min:", random_min)
if seed is not None:
  np.random.seed(seed)
  random.seed(seed)
  print("Seed:", seed)
if profile_model is not None:
  print("Profile model:", profile_model)
stage_plots = args.stage_plots or args.stage_plots_only
stage_plot_out_dir = Path(args.stage_plot_out_dir).resolve()


def _stage_tick_positions(n_stages: int) -> np.ndarray:
  tick_step = max(1, n_stages // 20)
  ticks = np.arange(1, n_stages + 1, tick_step, dtype=int)
  if ticks[-1] != n_stages:
    ticks = np.append(ticks, n_stages)
  return ticks


def plot_stage_profiles(
  comps_remote_ms: np.ndarray,
  stage_input_bytes: np.ndarray,
  out_dir: Path,
  stage_tag: str,
) -> tuple[Path, Path]:
  out_dir.mkdir(parents=True, exist_ok=True)
  n_stages = int(comps_remote_ms.size)
  stage_idx = np.arange(1, n_stages + 1, dtype=int)
  xticks = _stage_tick_positions(n_stages)

  sns.set_theme(style="whitegrid", font_scale=0.8)

  fig_latency, ax_latency = plt.subplots(figsize=(11, 3.5))
  ax_latency.plot(stage_idx, comps_remote_ms, marker="o", linewidth=1.2, markersize=2.2, color="tab:blue")
  ax_latency.set_title(f"Per-Stage Latency ({stage_tag})")
  ax_latency.set_xlabel("Stage Index")
  ax_latency.set_ylabel("Latency [ms]")
  ax_latency.set_xticks(xticks)
  ax_latency.grid(True, alpha=0.35)
  fig_latency.tight_layout()

  latency_path = out_dir / f"{stage_tag}_stage_latency_ms.png"
  fig_latency.savefig(latency_path, dpi=180, bbox_inches="tight")

  fig_size, ax_size = plt.subplots(figsize=(11, 3.5))
  input_size_mb = stage_input_bytes / (1024.0 * 1024.0)
  ax_size.plot(stage_idx, input_size_mb, marker="o", linewidth=1.2, markersize=2.2, color="tab:green")
  ax_size.set_title(f"Per-Stage Input Size ({stage_tag})")
  ax_size.set_xlabel("Stage Index")
  ax_size.set_ylabel("Input Size [MiB]")
  ax_size.set_xticks(xticks)
  ax_size.grid(True, alpha=0.35)
  fig_size.tight_layout()

  input_size_path = out_dir / f"{stage_tag}_stage_input_size_mib.png"
  fig_size.savefig(input_size_path, dpi=180, bbox_inches="tight")
  return latency_path, input_size_path

alphas = [alpha_min + period*i for i in range(int(round((alpha_max-alpha_min)/period))+1)]
algs = ["AutoNeuro", "DOPart", "Neuro", "Remote Only", "Local Only", "DOPart-R", "DOPart-DR", "DOPart-AR", "DOPart-DAR", "Threat Based", "OPT"]
print(alphas)
current_comps_remote, input_data_real = system_values(v, profile_model=profile_model)
current_comps_remote = np.asarray(current_comps_remote, dtype=float)
input_data_real = np.asarray(input_data_real, dtype=float)

print(len(current_comps_remote))

if stage_plots:
  if profile_model is not None:
    stage_tag = profile_model
  else:
    stage_tag = "resnet152" if v == 152 else "resnet200" if v == 200 else f"stages_{v}"
  latency_path, input_size_path = plot_stage_profiles(
    current_comps_remote,
    input_data_real,
    stage_plot_out_dir,
    stage_tag,
  )
  print(f"Saved stage latency plot: {latency_path}")
  print(f"Saved stage input-size plot: {input_size_path}")
  if args.stage_plots_only:
    if not args.no_show:
      plt.show()
    else:
      plt.close("all")
    raise SystemExit(0)

if comms_range_factor == 1:
  lb, ub = commsRange(alpha_min, alpha_max, log_uniform, alpha_fixed, mid_scale) 

def genAlphas(a,b,size):
    if not log_uniform:
        return np.random.uniform(a, b, size=size)
    else:
        return np.power(2.0, np.random.uniform(a, b, size=size))

def generateSamples(i):
  local_alpha_min = alpha_min
  local_alpha_max = alpha_max
  local_lb = lb
  local_ub = ub

  if alpha_fixed:
    local_alpha_max = alphas[i]
  else:
    local_alpha_min = alphas[i]

  if log_uniform:
    b = math.pow(2,local_alpha_max)
    a = math.pow(2,local_alpha_min)
  else:
    b = local_alpha_max
    a = local_alpha_min

  if comms_range_factor == 2:
    local_lb, local_ub = commsRange(alpha_min, alpha_max,log_uniform=log_uniform, alpha_fixed=alpha_fixed) 

  
  n_layers = current_comps_remote.size

  TALG = [np.zeros(NUM_SAMPLES, dtype=float) for _ in range(len(algs))]
  TOFF = [np.zeros(NUM_SAMPLES, dtype=float) for _ in range(len(algs))]

  alpha_scales = genAlphas(local_alpha_min, local_alpha_max, size=(NUM_SAMPLES, n_layers))
  current_comps_local = alpha_scales * current_comps_remote

  Rl = float(np.sum(current_comps_remote))
  bandwidth=input_data_real[0]/Rl

  if not comms_uniform:
    rb = bandwidth * local_lb + np.random.random((NUM_SAMPLES, n_layers)) * bandwidth * local_ub
    comms_body = input_data_real / rb
  else:
    comms_body = (a - 1 + np.random.random((NUM_SAMPLES, n_layers)) * (b - a)) * Rl
  current_comms_uniform = np.concatenate((comms_body, np.zeros((NUM_SAMPLES, 1), dtype=float)), axis=1)

  local_prefix = np.cumsum(current_comps_local, axis=1, dtype=float)
  local_prefix = np.concatenate((np.zeros((NUM_SAMPLES, 1), dtype=float), local_prefix), axis=1)
  remote_suffix = np.empty(n_layers + 1, dtype=float)
  remote_suffix[-1] = 0.0
  if n_layers:
    remote_suffix[:-1] = np.cumsum(current_comps_remote[::-1])[::-1]

  makespan_matrix = local_prefix + current_comms_uniform + remote_suffix
  if 0 not in IGNORED_ALGS:
    makespan = np.mean(makespan_matrix, axis=0)
    ANeuro_best_point = int(np.argmin(makespan))
  else:
    ANeuro_best_point = 0

  if random_min:
    a = min(a,1)

  ratio = TBP_ratio(a,b,current_comms_uniform.shape[1])
  rand_params = rand_threshold_params(a, b)

  for j in range(NUM_SAMPLES):
    comms_j = current_comms_uniform[j]
    local_j = current_comps_local[j]
    totals_j = makespan_matrix[j]

    opt_best = float(np.min(totals_j))
    opt_idx = int(np.argmin(totals_j))
    
    if 0 not in IGNORED_ALGS:
      TALG[0][j] = float(totals_j[ANeuro_best_point])
    TOFF[0][j] = float(ANeuro_best_point)
    if 2 not in IGNORED_ALGS:
      TALG[2][j] = float(totals_j[-1])
    TOFF[2][j] = float(n_layers)
    
    alg_best12, _, idx12, _ = DOPart(comms_j, local_j, current_comps_remote, a, b, 0)
    alg_best5, _, idx5 = DOPartRAND(comms_j, local_j, current_comps_remote, a, b, rand_params=rand_params) #USED
    alg_best7, _, idx7 = DOPartARAND(comms_j, local_j, current_comps_remote, a, b) #USED
    alg_best9, _, idx9 = TBP(comms_j, local_j, current_comps_remote, a, b, ratio) #USED

    if 6 not in IGNORED_ALGS:
      TALG[6][j], _, idx6 = DOPartRANDR(comms_j, local_j, current_comps_remote, a, b, rand_params=rand_params) #USED
      TOFF[6][j] = float(idx6)
    if 8 not in IGNORED_ALGS:
      TALG[8][j], _, idx8 = DOPartARANDR(comms_j, local_j, current_comps_remote, a, b, rand_params=rand_params) #USED
      TOFF[8][j] = float(idx8)
    
    TALG[1][j] = alg_best12
    TOFF[1][j] = float(idx12)
    TALG[3][j] = float(totals_j[0])
    TOFF[3][j] = 0.0
    TALG[4][j] = float(totals_j[-1])
    TOFF[4][j] = float(n_layers)
    TALG[5][j] = alg_best5
    TOFF[5][j] = float(idx5)
    TALG[7][j] = alg_best7
    TOFF[7][j] = float(idx7)
    TALG[9][j] = alg_best9
    TOFF[9][j] = float(idx9)
    TALG[10][j] = opt_best
    TOFF[10][j] = float(opt_idx)
  print("Alpha: ", alphas[i],"Average local computation delay:", makespan_matrix.mean(axis=0)[-1]/sum(current_comps_remote), "Average remote computation delay:", makespan_matrix.mean(axis=0)[0]/sum(current_comps_remote))
  return TALG, TOFF



# if v!=0:
#   log_uniform = False
#   comms_uniform = True
# else:
#   log_uniform = True
#   comms_uniform = False

TALG_final = [[] for _ in range(len(algs))]
TOFF_final = [[] for _ in range(len(algs))]

for i in range(len(alphas)):
  TALG, TOFF = generateSamples(i)
  print(f"Completed for alpha value = {alphas[i]}")
  for j in range(len(TALG)):
    TALG_final[j].append(TALG[j])
    TOFF_final[j].append(TOFF[j])

ignore = sorted(IGNORED_ALGS)
active_algs = [algs[k] for k in range(len(algs)) if k not in ignore]
compiled_frames = []
offload_frames = []
alpha_values = np.asarray(alphas, dtype=float)

for k in range(len(algs)):
  print("Processing Algorithm:",algs[k])
  if k not in ignore:
    values = np.asarray(TALG_final[k], dtype=float).reshape(-1)
    offload_idx = np.asarray(TOFF_final[k], dtype=float).reshape(-1)
    alpha_repeated = np.repeat(alpha_values, NUM_SAMPLES)
    compiled_frames.append(
      pd.DataFrame(
        {
          "Average Makespan": values,
          "Alpha": alpha_repeated,
          "Alg": algs[k],
        }
      )
    )
    offload_frames.append(
      pd.DataFrame(
        {
          "Offload Layer": offload_idx,
          "Alpha": alpha_repeated,
          "Alg": algs[k],
        }
      )
    )

df_main1 = pd.concat(compiled_frames, ignore_index=True)
df_main1["Average Makespan"] = df_main1["Average Makespan"].div(0.001)
df_offload = pd.concat(offload_frames, ignore_index=True)
sns.set(rc={'figure.figsize':(6,3)})
plt.rcParams["figure.figsize"] = [6,3]
plt.rcParams["figure.autolayout"] = True

#Lower Bandwidth range 3 Rl Rl = (r+1/8)Tr/2 Fixed comms and comps for neuro Autodidactic comms small

sns.set_theme(font_scale=0.7, style='white')

d_style = {alg: "" for alg in algs}
d_style[algs[-1]] = (5, 10)
palette = sns.color_palette("tab10", n_colors=max(1, len(active_algs)))


def _lineplot_with_ci(ax, lineplot_kwargs):
  ci_enabled = ci_level > 0
  try:
    if ci_enabled:
      return sns.lineplot(ax=ax, errorbar=("ci", ci_level), **lineplot_kwargs)
    return sns.lineplot(ax=ax, errorbar=None, **lineplot_kwargs)
  except TypeError:
    if ci_enabled:
      return sns.lineplot(ax=ax, ci=ci_level, **lineplot_kwargs)
    return sns.lineplot(ax=ax, ci=None, **lineplot_kwargs)


def _set_alpha_xlabel(h):
  if alpha_fixed:
    if log_uniform:
      h.set_xlabel(r'$\log_2\alpha$' + r'$_\mathregular{max}$')
    else:
      h.set_xlabel(r'$\alpha$' + r'$_\mathregular{max}$')
  else:
    if log_uniform:
      h.set_xlabel(r'$\log_2\alpha$' + r'$_\mathregular{min}$')
    else:
      h.set_xlabel(r'$\alpha$' + r'$_\mathregular{min}$')


def _style_legend(ax):
  handles, labels = ax.get_legend_handles_labels()
  if len(handles) >= 2:
    handles[0], handles[1] = handles[1], handles[0]
    labels[0], labels[1] = labels[1], labels[0]
  ax.legend(handles=handles, labels=labels)


# Main makespan plot.
fig_main, ax_main = plt.subplots()
lineplot_kwargs_main = dict(
  x="Alpha",
  y="Average Makespan",
  hue="Alg",
  data=df_main1,
  style="Alg",
  linewidth=1,
  hue_order=active_algs,
  style_order=active_algs,
  palette=palette,
  markers=True,
  dashes=d_style,
  markersize=8,
  seed=seed,
)
h_main = _lineplot_with_ci(ax_main, lineplot_kwargs_main)
h_main.set_xticks(alphas)
_set_alpha_xlabel(h_main)
h_main.set_ylabel(r'Average ' + r'$T_\mathregular{ALG}$' + r' [ms]')
h_main.ticklabel_format(useMathText=True)
_style_legend(ax_main)
ax_main.grid(True)
[x.set_linewidth(0.5) for x in ax_main.spines.values()]

# Add command-line parameters to the figures.
cmd_parts = [
  f"stages={v}",
  f"profile_model={profile_model}",
  f"comms_uniform={comms_uniform}",
  f"log_uniform={log_uniform}",
  f"alpha_min={alpha_min}",
  f"alpha_max={alpha_max}",
  f"alpha_fixed={alpha_fixed}",
  f"lower_bound={lb}",
  f"upper_bound={ub}",
  f"ci={ci_level}",
  f"seed={seed}",
  f"offload_plot={offload_plot}",
]
fig_main.text(
  0.01, 0.01,
  "Args: " + ", ".join(cmd_parts),
  ha="left",
  va="bottom",
  fontsize=7,
)

fig_offload = None
if offload_plot:
  fig_offload, ax_offload = plt.subplots()
  lineplot_kwargs_offload = dict(
    x="Alpha",
    y="Offload Layer",
    hue="Alg",
    data=df_offload,
    style="Alg",
    linewidth=1,
    hue_order=active_algs,
    style_order=active_algs,
    palette=palette,
    markers=True,
    dashes=d_style,
    markersize=8,
    seed=seed,
  )
  h_offload = _lineplot_with_ci(ax_offload, lineplot_kwargs_offload)
  h_offload.set_xticks(alphas)
  _set_alpha_xlabel(h_offload)
  h_offload.set_ylabel("Offload Layer Index")
  max_layer_idx = int(current_comps_remote.size)
  h_offload.set_ylim(-0.5, max_layer_idx + 0.5)
  ytick_step = max(1, max_layer_idx // 10)
  offload_ticks = np.arange(0, max_layer_idx + 1, ytick_step, dtype=int)
  if offload_ticks[-1] != max_layer_idx:
    offload_ticks = np.append(offload_ticks, max_layer_idx)
  h_offload.set_yticks(offload_ticks)
  _style_legend(ax_offload)
  ax_offload.grid(True)
  [x.set_linewidth(0.5) for x in ax_offload.spines.values()]
  fig_offload.text(
    0.01, 0.01,
    "Args: " + ", ".join(cmd_parts),
    ha="left",
    va="bottom",
    fontsize=7,
  )

ts = datetime.now().strftime("%Y-%b-%d_%H-%M-%S")
main_name = f"DOPart_Randomized_{ts}.pdf"
offload_name = f"DOPart_Randomized_Offload_{ts}.pdf"
out_dir = Path(fr"C:\Users\shiva\Dropbox\shared\DOPart\Randomized\Experiments")
try:
  out_dir.mkdir(parents=True, exist_ok=True)
  fig_main.savefig(out_dir / main_name, bbox_inches="tight")
  if fig_offload is not None:
    fig_offload.savefig(out_dir / offload_name, bbox_inches="tight")
except OSError:
  fig_main.savefig(Path.cwd() / main_name, bbox_inches="tight")
  if fig_offload is not None:
    fig_offload.savefig(Path.cwd() / offload_name, bbox_inches="tight")

if not args.no_show:
  plt.show()
else:
  plt.close("all")

#  python .\project\plot.py --stages 0 --no-comms-uniform --log-uniform --alpha-min 0 --alpha-max 2.5 --alpha-fixed --lower-bound 0.25 --upper-bound 2.5
