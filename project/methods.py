import random
import numpy as np
import math

from scipy.special import lambertw

e = math.e

def ALPHAOPT(current_comms_uniform,current_comps_local,current_comps_remote):

  makespans = []

  for i in range(len(current_comms_uniform)):
    makespans.append(sum(current_comps_local[:i])+current_comms_uniform[i]+sum(current_comps_remote[i:]))
  return np.min(makespans), np.argmin(makespans), np.max(makespans)/np.min(makespans)

def DOPart(current_comms_uniform,current_comps_local,current_comps_remote,alm,alM,counter):

  best = sum(current_comps_local)
  c_best = current_comms_uniform[-1]

  for i in range(len(current_comms_uniform)):
      if i == len(current_comps_remote):
        Pr_i = 0
      else:
        Pr_i = current_comps_remote[i]

      term0 = sum(current_comps_local[:i]) + sum(current_comps_remote[i:])
      term1 = sum(current_comps_local[:i]) + alm*Pr_i + min(alm,1)*sum(current_comps_remote[i+1:])
      term2 = sum(current_comps_local[:i]) + max(alM,1)*sum(current_comps_remote[i:])
 
      if current_comms_uniform[i] <= (math.sqrt(term1*term2) - term0):
          c_best = current_comms_uniform[i]
          best = term0 + c_best
          if i is not (len(current_comms_uniform)-1):
            counter = counter + 1
          return best, c_best, i, counter
  return best, c_best, len(current_comms_uniform), counter


#Non Adaptive Double Randomized Thresholding Algorithm
def DOPartRANDR(current_comms_uniform,current_comps_local,current_comps_remote,r,R):

  Tr = sum(current_comps_remote)
  alm = r
  alM = R
  al = alM/alm
  ep = (-R* lambertw((1/(al*e))-1/e)).real


  for i in range(len(current_comms_uniform)):
      
      b = random.random()
      bound = (-R/(R-ep))*np.log(ep/(R-r))
      if b >= bound:
        k = R
      else:
        k = R - (R-r)/(math.exp(b*(R-ep)/R))

      thresh = k
      
      T_i = sum(current_comps_local[:i]) + current_comms_uniform[i] + sum(current_comps_remote[i:])
      if  T_i <= thresh*Tr:
        return T_i, current_comms_uniform[i], i

  return sum(current_comps_local),0, len(current_comms_uniform)

#Non Adaptive Randomized Thresholding Algorithm
def DOPartRAND(current_comms_uniform,current_comps_local,current_comps_remote,r,R):

  Tr = sum(current_comps_remote)
  alm = r
  alM = R
  al = alM/alm
  ep = (-R* lambertw((1/(al*e))-1/e)).real
  b = random.random()

  bound = (-R/(R-ep))*np.log(ep/(R-r))

  if b >= bound:
    k = R
  else:
    k = R - (R-r)/(math.exp(b*(R-ep)/R))

  thresh = k

  for i in range(len(current_comms_uniform)):
      T_i = sum(current_comps_local[:i]) + current_comms_uniform[i] + sum(current_comps_remote[i:])
      if  T_i <= thresh*Tr:
        return T_i, current_comms_uniform[i], i

  return sum(current_comps_local),0, len(current_comms_uniform)

#Adaptive Randomized Thresholding Algorithm
def DOPartARAND(current_comms_uniform,current_comps_local,current_comps_remote,r,R):

  
  b = random.random()
  for i in range(len(current_comms_uniform)):
      
      T_bar = sum(current_comps_local[:i]) + sum(current_comps_remote[i:])
      alm = sum(current_comps_local[:i]) + r*sum(current_comps_remote[i:])
      alm = alm/T_bar
      alM = sum(current_comps_local[:i]) + R*sum(current_comps_remote[i:])
      alM = alM/T_bar
      al = alM/alm

      ep = (-alM* lambertw((1/(al*e))-1/e)).real

      bound = (-alM/(alM-ep))*np.log(ep/(alM-alm))

      if b >= bound:
        k = alM
      else:
        k = alM - (alM-alm)/(math.exp(b*(alM-ep)/alM))

      thresh = k
      T_i = sum(current_comps_local[:i]) + current_comms_uniform[i] + sum(current_comps_remote[i:])
      if  T_i <= thresh*T_bar:
        return T_i, current_comms_uniform[i], i

  return sum(current_comps_local),0, len(current_comms_uniform)

#Non Adaptive Double Randomized Thresholding Algorithm
def DOPartARANDR(current_comms_uniform,current_comps_local,current_comps_remote,r,R):

  Tr = sum(current_comps_remote)
  alm = r
  alM = R
  al = alM/alm
  ep = (-R* lambertw((1/(al*e))-1/e)).real


  for i in range(len(current_comms_uniform)):
      
      b = random.random()
      bound = (-R/(R-ep))*np.log(ep/(R-r))
      if b >= bound:
        k = R
      else:
        k = R - (R-r)/(math.exp(b*(R-ep)/R))

      thresh = k
      
      T_i = sum(current_comps_local[:i]) + current_comms_uniform[i] + sum(current_comps_remote[i:])
      if  T_i <= thresh*Tr:
        return T_i, current_comms_uniform[i], i

  return sum(current_comps_local),0, len(current_comms_uniform)

def TBP_ratio(r,R,n):
  print("R: ", R, "r: ", r)
  r = min(r,1)

  if r == R:
     return 1

  diff = R-r
  ratio_tbp = R
  for i in range(10000):
      x = random.uniform(1,R/r)
      t = math.pow(((1-1/x)/(1-r/R)),(1/n))
      v = 1/(n*(1-t))
      if abs(v-x) < diff:
          diff = abs(v-x)
          ratio_tbp = x
  return ratio_tbp


#Threat-Based Policy (Randomized)
def TBP(current_comms_uniform,current_comps_local,current_comps_remote,r,R,ratio):

  Tr = sum(current_comps_remote)
  alM = R
  alm = r
  al = alM/alm
  # old_i = alM*Tr/ratio #empirically better than old for smaller values
  old_i = alM*Tr #Original Plots
  # old_i = ratio*Tr #experimenting
  a = 0.0
  # ratio = 1/(1+(lambertw((1/(al*e))-1/e).real))

  for i in range(len(current_comms_uniform)):
      T_i = sum(current_comps_local[:i]) + current_comms_uniform[i] + sum(current_comps_remote[i:])
      if T_i < old_i:
          denom = (alM*Tr - T_i)
          if denom == 0:
              continue

          if a == 0:
              s = (alM*Tr - ratio*T_i)/denom
          else:
              s = ratio*(old_i - T_i)/denom

          old_i = T_i
          a += s

          if random.random() <= s:
              return T_i, current_comms_uniform[i], i

  return sum(current_comps_local),0, len(current_comms_uniform)