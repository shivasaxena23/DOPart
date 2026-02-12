from datetime import datetime
import math
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data_generation import system_values
from methods import TBP, DOPart, DOPartARAND, DOPartARANDR, DOPartRAND, DOPartRANDR, TBP_ratio, rand_threshold_params
import argparse

from findCommsRange import commsRange

parser = argparse.ArgumentParser()
parser.add_argument("--stages", type=int, default=0)

# booleans
parser.add_argument("--comms-uniform", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--log-uniform", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--random-min", action=argparse.BooleanOptionalAction, default=False)

# floats
parser.add_argument("--alpha-min", type=float, default=1.0)
parser.add_argument("--alpha-max", type=float, default=4.0)
parser.add_argument("--period", type=float, default=0.5)
parser.add_argument("--alpha-fixed", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--comms-range-factor", type=int, default=1)
parser.add_argument("--lower-bound", type=float, default=0.25)
parser.add_argument("--upper-bound", type=float, default=2.5)

args = parser.parse_args()
NUM_SAMPLES = 7000
IGNORED_ALGS = {0, 2, 6, 8}

v = args.stages
comms_uniform = args.comms_uniform
lb = args.lower_bound
ub = args.upper_bound
comms_range_factor = args.comms_range_factor
log_uniform = args.log_uniform
alpha_fixed = args.alpha_fixed
alpha_min = args.alpha_min
alpha_max = args.alpha_max
period = args.period
random_min = args.random_min

alphas = [alpha_min + period*i for i in range(int(round((alpha_max-alpha_min)/period))+1)]
algs = ["AutoNeuro", "DOPart", "Neuro", "Remote Only", "Local Only", "DOPart-R", "DOPart-DR", "DOPart-AR", "DOPart-DAR", "Threat Based", "OPT"]
print(alphas)
current_comps_remote, input_data_real = system_values(v)
current_comps_remote = np.asarray(current_comps_remote, dtype=float)
input_data_real = np.asarray(input_data_real, dtype=float)

print(len(current_comps_remote))

if comms_range_factor == 1:
  lb, ub = commsRange(alpha_min, alpha_max,log_uniform=log_uniform, alpha_fixed=alpha_fixed) 

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
    
    if 0 not in IGNORED_ALGS:
      TALG[0][j] = float(totals_j[ANeuro_best_point])
    if 2 not in IGNORED_ALGS:
      TALG[2][j] = float(totals_j[-1])
    
    alg_best12, _, _, _ = DOPart(comms_j, local_j, current_comps_remote, a, b, 0)
    alg_best5, _, _ = DOPartRAND(comms_j, local_j, current_comps_remote, a, b, rand_params=rand_params) #USED
    alg_best7, _, _ = DOPartARAND(comms_j, local_j, current_comps_remote, a, b) #USED
    alg_best9, _, _ = TBP(comms_j, local_j, current_comps_remote, a, b, ratio) #USED

    if 6 not in IGNORED_ALGS:
      TALG[6][j], _, _ = DOPartRANDR(comms_j, local_j, current_comps_remote, a, b, rand_params=rand_params) #USED
    if 8 not in IGNORED_ALGS:
      TALG[8][j], _, _ = DOPartARANDR(comms_j, local_j, current_comps_remote, a, b, rand_params=rand_params) #USED
    
    TALG[1][j] = alg_best12
    TALG[3][j] = float(totals_j[0])
    TALG[4][j] = float(totals_j[-1])
    TALG[5][j] = alg_best5
    TALG[7][j] = alg_best7
    TALG[9][j] = alg_best9
    TALG[10][j] = opt_best
  print("Alpha: ", alphas[i],"Average local computation delay:", makespan_matrix.mean(axis=0)[-1]/sum(current_comps_remote), "Average remote computation delay:", makespan_matrix.mean(axis=0)[0]/sum(current_comps_remote))
  return (TALG)



# if v!=0:
#   log_uniform = False
#   comms_uniform = True
# else:
#   log_uniform = True
#   comms_uniform = False

TALG_final = [[] for _ in range(len(algs))] 

for i in range(len(alphas)):
  TALG = generateSamples(i)
  print(f"Completed for alpha value = {alphas[i]}")
  for j in range(len(TALG)):
    TALG_final[j].append(TALG[j])

ignore = sorted(IGNORED_ALGS)
compiled_frames = []
alpha_values = np.asarray(alphas, dtype=float)

for k in range(len(algs)):
  print("Processing Algorithm:",algs[k])
  if k not in ignore:
    values = np.asarray(TALG_final[k], dtype=float).reshape(-1)
    compiled_frames.append(
      pd.DataFrame(
        {
          "Average Makespan": values,
          "Alpha": np.repeat(alpha_values, NUM_SAMPLES),
          "Alg": algs[k],
        }
      )
    )

df_main1 = pd.concat(compiled_frames, ignore_index=True)
df_main1["Average Makespan"] = df_main1["Average Makespan"].div(0.001)
sns.set(rc={'figure.figsize':(6,3)})
plt.rcParams["figure.figsize"] = [6,3]
plt.rcParams["figure.autolayout"] = True

#Lower Bandwidth range 3 Rl Rl = (r+1/8)Tr/2 Fixed comms and comps for neuro Autodidactic comms small

sns.set_theme(font_scale=0.7, style='white')

fig, ax = plt.subplots()


d_style = {}
for i in algs:
  d_style[i]=''

d_style[algs[-1]] = (5, 10)

lineplot_kwargs = dict(
  x="Alpha",
  y="Average Makespan",
  hue="Alg",
  data=df_main1,
  style="Alg",
  linewidth=1,
  palette=['g', 'black','b','r','magenta', 'orange', 'cyan'],
  markers=True,
  dashes=d_style,
  markersize=8,
)
try:
  h = sns.lineplot(errorbar=None, **lineplot_kwargs)
except TypeError:
  h = sns.lineplot(ci=None, **lineplot_kwargs)
h.set_xticks(alphas) # <--- set the ticks first

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

h.set_ylabel(r'Average ' + r'$T_\mathregular{ALG}$' + r' [ms]')
h.ticklabel_format(useMathText=True)

# Add command-line parameters to the figure.
cmd_parts = [
  f"stages={v}",
  f"comms_uniform={comms_uniform}",
  f"log_uniform={log_uniform}",
  f"alpha_min={alpha_min}",
  f"alpha_max={alpha_max}",
  f"alpha_fixed={alpha_fixed}",
  f"lower_bound={lb}",
  f"upper_bound={ub}",
]
fig.text(
  0.01, 0.01,
  "Args: " + ", ".join(cmd_parts),
  ha="left",
  va="bottom",
  fontsize=7,
)

handles, labels = ax.get_legend_handles_labels()
handles[0], handles[1] = handles[1], handles[0]
labels[0], labels[1] = labels[1], labels[0]
ax.legend(handles=handles[0:], labels=labels[0:])
ax.grid(True)

[x.set_linewidth(0.5) for x in ax.spines.values()]

ts = datetime.now().strftime("%Y-%b-%d_%H-%M-%S")
out_path = Path(
  fr"C:\Users\shiva\Dropbox\shared\DOPart\Randomized\Experiments\DOPart_Randomized_{ts}.pdf"
)
try:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(out_path, bbox_inches="tight")
except OSError:
  fallback_path = Path.cwd() / f"DOPart_Randomized_{ts}.pdf"
  plt.savefig(fallback_path, bbox_inches="tight")

plt.show()

#  python .\project\plot.py --stages 0 --no-comms-uniform --log-uniform --alpha-min 0 --alpha-max 2.5 --alpha-fixed --lower-bound 0.25 --upper-bound 2.5
