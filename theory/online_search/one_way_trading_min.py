import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random
from scipy.special import lambertw
from math import e

n = 100
np.random.seed(42)
m = 1
M_set = [1.5,2,4,6,8,10,12] 

# RAND does better than DET and RANDR for small ratios and small n or all ratios and large n
# As the number of days/layers/n increase the randomized does better than DET and even RANDR

# RAND<RANDR<DET for small ratios, but for larger ratios, DET<RANDR<RAND (More evident for larger n)

def RAND(A,M):
  
  b = random.random()
  al = M/m
  ep = (-M* lambertw((1/(al*e))-1/e).real)
  bound = (-M/(M-ep))*np.log(ep/(M-m))
  # print("RAND bound:", bound)
  for i in A[:-1]:



      k = M - (M-m)/(math.exp(b*(M-ep)/M))

      thresh = k

      if  i <= thresh:
        # print(T_i, alg)
        return i

  # print(sum(current_comps_local), alg)
  return A[-1]



def RANDR(A,M):
  
  al = M/m
  ep = (-M* lambertw((1/(al*e))-1/e).real)
  bound = (-M/(M-ep))*np.log(ep/(M-m))
  # print("RANDR bound:", bound)
  for i in A[:-1]:
      b = random.random()


      k = M - (M-m)/(math.exp(b*(M-ep)/M))

      thresh = k

      if  i <= thresh:
        # print(T_i, alg)
        return i

  # print(sum(current_comps_local), alg)
  return A[-1]

def DET(A,M):

  for i in A[:-1]:
      thresh = math.sqrt(M * m)

      if  i <= thresh:
        # print(T_i, alg)
        return i

  # print(sum(current_comps_local), alg)
  return A[-1]

#Non Adaptive Randomized Thresholding Algorithm

def THREATR(A,M):

  al = M/m
  ratio = 1/(1+(lambertw((1/(al*e))-1/e).real))
  old_i = M/ratio
  a = 0
  for i in A[:-1]:
      if  i < old_i:
        if a == 0:
           s = (M-ratio*i)/(M-i)
        else:
           s = ratio*(old_i-i)/(M-i)
        old_i = i
        a = a + s
        if random.random() <= s:
            return i
  return A[-1]

def THREATD(A,M):

  al = M/m
  ratio = 1/(1+(lambertw((1/(al*e))-1/e).real))
  old_i = M/ratio
  a = 0
  total = 0
  for i in A[:-1]:
      if  i < old_i:
        if a == 0:
           s = (M-ratio*i)/(M-i)
        else:
           s = ratio*(old_i-i)/(M-i)
        old_i = i
        a = a + s
        total += s*old_i
  total += A[-1] * (1 - a)
  return total


for M in M_set:
    TOPT_sum = 0
    TDET_sum = 0
    TRAND_sum = 0
    TRANDR_sum = 0
    TTHREATR_sum = 0
    TTHREATD_sum = 0

    for  k in range(10000):
        vals = []
        for i in range(n):
          val = m +random.random() * (M - m)
          vals.append(val)
        TOPT = min(vals)
        TDET = DET(vals,M)
        TRAND = RAND(vals,M)
        TRANDR = RANDR(vals,M)
        TTHREATR = THREATR(vals,M)
        TTHREATD = THREATD(vals,M)
        TOPT_sum += TOPT
        TDET_sum += TDET    
        TRAND_sum += TRAND
        TRANDR_sum += TRANDR
        TTHREATR_sum += TTHREATR
        TTHREATD_sum += TTHREATD

    print("M:", M," TOPT:", TOPT_sum/10000, "TDET:", TDET_sum/10000, "TRAND:", TRAND_sum/10000, "TRANDR:", TRANDR_sum/10000, "TTHREATR:", TTHREATR_sum/10000, "TTHREATD:", TTHREATD_sum/10000)