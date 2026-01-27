import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random
from scipy.special import lambertw
from math import e

n = 50
np.random.seed(42)
m = 1
M_set = [2,3,4,8,16,32] 

def RAND(A,M,Tr):

  b = random.random()
  al = M/m
  ep = (-M* lambertw((1/(al*e))-1/e).real)
  bound = (-M/(M-ep))*np.log(ep/(M-m))

  for i in A[:-1]:


      if b >= bound:
        k = M
      else:
        k = M - (M-m)/(math.exp(b*(M-ep)/M))

      thresh = k

      if  i <= thresh*Tr:
        # print("RAND:", i)
        return i

  # print(sum(current_comps_local), alg)
  return A[-1]



def RANDR(A,M,Tr):
  
  al = M/m
  ep = (-M* lambertw((1/(al*e))-1/e).real)
  bound = (-M/(M-ep))*np.log(ep/(M-m))
  
  for i in A[:-1]:
      b = random.random()

      if b >= bound:
        k = M
      else:
        k = M - (M-m)/(math.exp(b*(M-ep)/M))

      thresh = k

      if  i <= thresh*Tr:
        # print("RANDR:", i)
        return i

  # print(sum(current_comps_local), alg)
  return A[-1]

def DET(A,M,Tr):

  for i in A[:-1]:
      thresh = math.sqrt(M * m)*Tr

      if  i <= thresh:
        # print("DET:", i)
        return i
  # print(sum(current_comps_local), alg)
  return A[-1]

#Non Adaptive Randomized Thresholding Algorithm

def THREATR(A,M,Tr):

  al = M/m
  ratio = 1/(1+(lambertw((1/(al*e))-1/e).real))
  old_i = M*Tr/ratio
  a = 0
  for i in A[:-1]:
      if  i < old_i:
        if a == 0:
           s = (M*Tr-ratio*i)/(M*Tr-i)
        else:
           s = ratio*(old_i-i)/(M*Tr-i)
        old_i = i
        a = a + s
        if random.random() <= s:
            return i
  return A[-1]

def THREATD(A,M,Tr):

  al = M/m
  ratio = 1/(1+(lambertw((1/(al*e))-1/e).real))
  old_i = M*Tr/ratio
  a = 0
  total = 0
  for i in A[:-1]:
      if  i < old_i:
        if a == 0:
           s = (M*Tr-ratio*i)/(M*Tr-i)
        else:
           s = ratio*(old_i-i)/(M*Tr-i)
        old_i = i
        a = a + s
        total += s*old_i
  total += A[-1] * (1 - a)
  return total

opt_offload = []

index = 0

for M in M_set:
    TOPT_sum = 0
    TDET_sum = 0
    TRAND_sum = 0
    TRANDR_sum = 0
    TTHREATR_sum = 0
    TTHREATD_sum = 0

    opt_offload_freq = [0 for _ in range(n+1)]
    for  k in range(10000):
        vals = []
        remote = [(0.5 +random.random()) for _ in range(n)]
        local = [(m +random.random() * (M - m))*remote[i] for i in range(n)]
        comm = [(m +random.random() * (M - m))*(((M+m))*6) for i in range(n)]

        for i in range(n):
          val = sum(local[:i]) + comm[i] + sum(remote[i:])
          vals.append(val)
        
        vals.append(sum(local))

        TOPT = min(vals)   
        TDET = DET(vals,M,sum(remote))
        TRAND = RAND(vals,M,sum(remote))
        TRANDR = RANDR(vals,M,sum(remote))
        TTHREATR = THREATR(vals,M,sum(remote))
        TTHREATD = THREATD(vals,M,sum(remote))
        TOPT_sum += TOPT
        TDET_sum += TDET    
        TRAND_sum += TRAND
        TRANDR_sum += TRANDR
        TTHREATR_sum += TTHREATR
        TTHREATD_sum += TTHREATD
        opt_offload_freq[np.argmin(vals)] += 1
    opt_offload.append(opt_offload_freq)
    print("M:", M," TOPT:", TOPT_sum/10000, "TDET:", TDET_sum/10000, "TRAND:", TRAND_sum/10000, "TRANDR:", TRANDR_sum/10000, "TTHREATR:", TTHREATR_sum/10000, "TTHREATD:", TTHREATD_sum/10000)
df = pd.DataFrame(opt_offload)
print(df)
sns.lineplot(data=df.T)
plt.show()
