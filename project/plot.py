import math
import random
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data_generation import system_values
from methods import ALPHAOPT, TBP, DOPart, DOPartARAND, DOPartARANDR, DOPartRAND, DOPartRANDR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--layers", type=int, default=0)
args = parser.parse_args()

v = args.layers

alphas = [1,1.5,2,2.5,3]

algs = ["AutoNeuro", "DOPart", "Neuro", "Remote Only", "Local Only", "DOPart-R", "DOPart-DR", "DOPart-AR", "DOPart-DAR", "Threat Based", "OPT"]

current_comps_remote, input_data_real = system_values(v)


def genAlphas(a,b,n):
    return [a+(b-a)*random.random() for _ in range(n)]

def generateSamples(i):

  R = alphas[i]
  r = 1
  if log_uniform == True:
    b = math.pow(2,R)
    a = math.pow(2,r)
  else:
     b = R
     a = r

  lb = 0.25
  ub = 2.5
  

  TALG = [[] for _ in range(len(algs))] 
  


  current_comps_local = []
  for z in range(7000):
    current_comps_local.append(np.multiply(current_comps_remote,genAlphas(a,b,len(current_comps_remote))))

  current_comms_uniform = []
  Rl = sum(current_comps_remote)
  bandwidth=input_data_real[0]/Rl
  for z in range(7000):
    current_comms_uniform_AN = []
    for p in range(len(current_comps_remote)):    
      rb = bandwidth*(lb) + random.random()*(bandwidth)*(ub)
      if comms_uniform == False:
        current_comms_uniform_AN.append(input_data_real[p]/rb)
      else:
        current_comms_uniform_AN.append((a-1+random.random()*(b-a))*Rl)
    current_comms_uniform_AN.append(0)
    current_comms_uniform.append(current_comms_uniform_AN)

  makespan = []
  max_makespan = []
  min_makespan = []
  for p in range(len(current_comms_uniform[0])):
      val = 0
      max = 0
      min = 100*sum(current_comps_remote)
      for z in range(7000):
          val = val + sum(current_comps_local[z][:p])+current_comms_uniform[z][p]+sum(current_comps_remote[p:])
          if sum(current_comps_local[z][:p])+current_comms_uniform[z][p]+sum(current_comps_remote[p:]) > max:
            max = sum(current_comps_local[z][:p])+current_comms_uniform[z][p]+sum(current_comps_remote[p:])
          if sum(current_comps_local[z][:p])+current_comms_uniform[z][p]+sum(current_comps_remote[p:]) < min:
            min = sum(current_comps_local[z][:p])+current_comms_uniform[z][p]+sum(current_comps_remote[p:])
      val = val/7000
      makespan.append(val)
      max_makespan.append(max)
      min_makespan.append(min)
  ANeuro_best_point = np.argmin(makespan)
  for j in range(7000):

    alg_best11 = sum(current_comps_local[j][:ANeuro_best_point]) + current_comms_uniform[j][ANeuro_best_point] + sum(current_comps_remote[ANeuro_best_point:])
    
    comm_neuro = 0
    comp_neuro = current_comps_local[j][0]/current_comps_remote[0]
    current_comms_uniform_neuro = [input_data_real[p]/comm_neuro for p in range(len(current_comps_local[j]))]
    current_comms_uniform_neuro.append(0)
    current_comps_local_neuro = comp_neuro*current_comps_remote
    alg_best13, opt_best_point, max_seq = ALPHAOPT(current_comms_uniform_neuro,current_comps_local_neuro,current_comps_remote)
    alg_best13 = sum(current_comps_local[j][:opt_best_point]) + current_comms_uniform[j][opt_best_point] + sum(current_comps_remote[opt_best_point:])
    
    opt_best, opt_best_point, max_seq = ALPHAOPT(current_comms_uniform[j],current_comps_local[j],current_comps_remote)
    
    alg_best12, c_cut12, alg_best2_point, counter = DOPart(current_comms_uniform[j],current_comps_local[j],current_comps_remote, a, b,0)
    alg_best5, c_cut5, alg_best5_point = DOPartRAND(current_comms_uniform[j],current_comps_local[j],current_comps_remote, a, b) #USED
    alg_best6, c_cut6, alg_best6_point = DOPartRANDR(current_comms_uniform[j],current_comps_local[j],current_comps_remote, a, b) #USED
    alg_best7, c_cut7, alg_best7_point = DOPartARAND(current_comms_uniform[j],current_comps_local[j],current_comps_remote, a, b) #USED
    alg_best8, c_cut8, alg_best8_point = DOPartARANDR(current_comms_uniform[j],current_comps_local[j],current_comps_remote, a, b) #USED
    alg_best9, c_cut9, alg_best9_point = TBP(current_comms_uniform[j],current_comps_local[j],current_comps_remote, a, b) #USED
    
    alg_best2 = sum(current_comps_local[j][:0]) + current_comms_uniform[j][0] + sum(current_comps_remote[0:])
    alg_best3 = sum(current_comps_local[j][:len(current_comps_local[j])]) + current_comms_uniform[j][len(current_comps_local[j])] + sum(current_comps_remote[len(current_comps_local[j]):])


    TALG[0].append(alg_best11)
    TALG[1].append(alg_best12)
    TALG[2].append(alg_best13)
    TALG[3].append(alg_best2)
    TALG[4].append(alg_best3)
    TALG[5].append(alg_best5)
    TALG[6].append(alg_best6)
    TALG[7].append(alg_best7)
    TALG[8].append(alg_best8)
    TALG[9].append(alg_best9)
    TALG[10].append(opt_best)
  return (TALG)

log_uniform = False
comms_uniform = True
TALG_final = [[] for _ in range(len(algs))] 

for i in range(len(alphas)):
  TALG = generateSamples(i)
  print(f"Completed for alpha value = {alphas[i]}")
  for j in range(len(TALG)):
    TALG_final[j].append(TALG[j])

ignore = [0,2]
compiled = []

for k in range(len(algs)):
  print("Processing Algorithm:",algs[k])
  if k not in ignore:
    for i in range(len(alphas)):
        for j in range(7000):
          compiled.append([TALG_final[k][i][j], alphas[i], algs[k]])

df_main1 = pd.DataFrame(compiled, columns = ['Average Makespan', 'Alpha', 'Alg'])
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

h = sns.lineplot(x="Alpha",y="Average Makespan", hue="Alg", data=df_main1,style="Alg",linewidth=1, palette=['g', 'black','b','r','magenta', 'orange', 'cyan'],
    markers=True, dashes=d_style, markersize=8, err_style="band", err_kws={'alpha':0.1}) #NEWDOPartRAND
h.set_xticks(alphas) # <--- set the ticks first
if log_uniform == True:
  h.set_xlabel(r'$\log_2\alpha$' + r'$_\mathregular{max}$')
else:
  h.set_xlabel(r'$\alpha$' + r'$_\mathregular{max}$')
h.set_ylabel(r'Average ' + r'$T_\mathregular{ALG}$' + r' [ms]')
h.ticklabel_format(useMathText=True)

handles, labels = ax.get_legend_handles_labels()
handles[0], handles[1] = handles[1], handles[0]
labels[0], labels[1] = labels[1], labels[0]
ax.legend(handles=handles[0:], labels=labels[0:])

[x.set_linewidth(0.5) for x in ax.spines.values()]

plt.savefig("resnet34_max.pdf", bbox_inches='tight')
plt.show()