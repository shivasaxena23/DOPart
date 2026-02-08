
import math
import random
from scipy.special import lambertw

e = math.e

R = 4
r = 1
n = 4

al = R/r

ratio_dp = 1/(1+(lambertw((1/(al*e))-1/e).real))

diff = R-r
ratio_tbp = R
for i in range(10000):
    x = random.uniform(r,R)
    t = math.pow(((1-1/x)/(1-r/R)),(1/n))
    v = 1/(n*(1-t))
    if abs(v-x) < diff:
        diff = abs(v-x)
        ratio_tbp = x

print("DOPart-R: ", ratio_dp,"TBP: ", ratio_tbp)