import random
import numpy as np
import math

from scipy.special import lambertw

e = math.e

def DOPartRandRatio(r, R):
  al = R / r
  return 1 / (1 + (lambertw((1 / (al * e)) - 1 / e).real))

def _prepare_cumsums(current_comms_uniform, current_comps_local, current_comps_remote):
  comms = np.asarray(current_comms_uniform, dtype=float)
  local = np.asarray(current_comps_local, dtype=float)
  remote = np.asarray(current_comps_remote, dtype=float)

  local_prefix = np.empty(local.size + 1, dtype=float)
  local_prefix[0] = 0.0
  if local.size:
    np.cumsum(local, out=local_prefix[1:])

  remote_suffix = np.empty(remote.size + 1, dtype=float)
  remote_suffix[-1] = 0.0
  if remote.size:
    remote_suffix[:-1] = np.cumsum(remote[::-1])[::-1]

  span = min(comms.size, local_prefix.size, remote_suffix.size)
  return comms, local, remote, local_prefix, remote_suffix, span


def _rand_threshold_params(r, R):
  if R == r:
    return float(R), None

  al = R / r
  ep = float((-R * lambertw((1 / (al * e)) - (1 / e))).real)
  bound = float((-R / (R - ep)) * math.log(ep / (R - r)))
  return ep, bound


def _sample_threshold(r, R, ep, bound, b):
  if bound is None or b >= bound:
    return float(R)
  return float(R - (R - r) / math.exp(b * (R - ep) / R))


def rand_threshold_params(r, R):
  return _rand_threshold_params(r, R)


def ALPHAOPT(current_comms_uniform, current_comps_local, current_comps_remote):
  comms, _, _, local_prefix, remote_suffix, span = _prepare_cumsums(
      current_comms_uniform, current_comps_local, current_comps_remote
  )

  makespans = local_prefix[:span] + comms[:span] + remote_suffix[:span]
  best_idx = int(np.argmin(makespans))
  best = float(makespans[best_idx])
  worst = float(np.max(makespans))
  ratio = worst / best if best != 0 else math.inf
  return best, best_idx, ratio


def DOPart(current_comms_uniform, current_comps_local, current_comps_remote, alm, alM, counter):
  comms, _, remote, local_prefix, remote_suffix, span = _prepare_cumsums(
      current_comms_uniform, current_comps_local, current_comps_remote
  )

  best = float(local_prefix[-1])
  c_best = float(comms[-1])
  min_alm = min(alm, 1.0)
  max_alM = max(alM, 1.0)

  for i in range(span):
      prefix_i = local_prefix[i]
      suffix_i = remote_suffix[i]
      pr_i = remote[i] if i < remote.size else 0.0
      suffix_next = remote_suffix[i + 1] if (i + 1) < remote_suffix.size else 0.0

      term0 = prefix_i + suffix_i
      term1 = prefix_i + alm * pr_i + min_alm * suffix_next
      term2 = prefix_i + max_alM * suffix_i

      if comms[i] <= (math.sqrt(term1 * term2) - term0):
          c_best = float(comms[i])
          best = float(term0 + c_best)
          if i != (comms.size - 1):
            counter += 1
          return best, c_best, i, counter
  return best, c_best, int(comms.size), counter


#Non Adaptive Double Randomized Thresholding Algorithm
def DOPartRANDR(current_comms_uniform, current_comps_local, current_comps_remote, r, R, rand_params=None):
  comms, _, _, local_prefix, remote_suffix, span = _prepare_cumsums(
      current_comms_uniform, current_comps_local, current_comps_remote
  )

  Tr = remote_suffix[0]
  ep, bound = rand_params if rand_params is not None else _rand_threshold_params(r, R)

  for i in range(span):
      b = random.random()
      thresh = _sample_threshold(r, R, ep, bound, b)
      T_i = local_prefix[i] + comms[i] + remote_suffix[i]
      if T_i <= thresh * Tr:
        return float(T_i), float(comms[i]), i

  return float(local_prefix[-1]), 0.0, int(comms.size)

#Non Adaptive Randomized Thresholding Algorithm
def DOPartRAND(current_comms_uniform, current_comps_local, current_comps_remote, r, R, rand_params=None):
  comms, _, _, local_prefix, remote_suffix, span = _prepare_cumsums(
      current_comms_uniform, current_comps_local, current_comps_remote
  )

  ratio = DOPartRandRatio(r, R)
  Tr = remote_suffix[0]
  ep, bound = rand_params if rand_params is not None else _rand_threshold_params(r, R)
  b = random.random()
  thresh = _sample_threshold(r, R, ep, bound, b)


  for i in range(span):
      T_i = local_prefix[i] + comms[i] + remote_suffix[i]
      if T_i <= thresh * Tr: # and T_i <= R*Tr/ratio
        return float(T_i), float(comms[i]), i

  return float(local_prefix[-1]), 0.0, int(comms.size)

#Adaptive Randomized Thresholding Algorithm
def DOPartARAND(current_comms_uniform, current_comps_local, current_comps_remote, r, R):
  comms, _, _, local_prefix, remote_suffix, span = _prepare_cumsums(
      current_comms_uniform, current_comps_local, current_comps_remote
  )

  b = random.random()
  for i in range(span):
      prefix_i = local_prefix[i]
      suffix_i = remote_suffix[i]
      T_bar = prefix_i + suffix_i
      if T_bar == 0:
        continue

      alm = (prefix_i + r * suffix_i) / T_bar
      alM = (prefix_i + R * suffix_i) / T_bar
      
      ratio = DOPartRandRatio(alm, alM)
      
      if alM == alm:
        thresh = alM
      else:
        al = alM / alm
        ep = float((-alM * lambertw((1 / (al * e)) - (1 / e))).real)
        bound = float((-alM / (alM - ep)) * math.log(ep / (alM - alm)))
        if b >= bound:
          thresh = alM
        else:
          thresh = float(alM - (alM - alm) / math.exp(b * (alM - ep) / alM))

      T_i = prefix_i + comms[i] + suffix_i
      if T_i <= thresh * T_bar and b <= bound: # and T_i <= alM * T_bar / ratio
        return float(T_i), float(comms[i]), i

  return float(local_prefix[-1]), 0.0, int(comms.size)

#Non Adaptive Double Randomized Thresholding Algorithm
def DOPartARANDR(current_comms_uniform, current_comps_local, current_comps_remote, r, R, rand_params=None):
  return DOPartRANDR(
      current_comms_uniform,
      current_comps_local,
      current_comps_remote,
      r,
      R,
      rand_params=rand_params,
  )

def TBP_ratio(r,R,n):
  print("r: ", r, "R: ", R)
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
  comms, _, _, local_prefix, remote_suffix, span = _prepare_cumsums(
      current_comms_uniform, current_comps_local, current_comps_remote
  )

  Tr = remote_suffix[0]
  alM = R
  # old_i = alM*Tr #Original Plots
  old_i = alM*Tr/ratio #Original Plots
  a = 0.0

  for i in range(span):
      T_i = local_prefix[i] + comms[i] + remote_suffix[i]
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
              return float(T_i), float(comms[i]), i

  return float(local_prefix[-1]), 0.0, int(comms.size)
