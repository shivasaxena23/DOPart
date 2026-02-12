import math
import random

f = lambda r,R: (math.pow(2,R)-math.pow(2,r))/(math.log(2)*(R-r))
g = lambda l, u: 1+ math.log(1+u/l)/(u)


def commsRange(r,R,log_uniform=True, alpha_fixed=False):

    if alpha_fixed:
        r, R = R, r
    if log_uniform:
        print(r,R)
        if r == R:
            mid = math.pow(2,R)
        else:
            mid = (f(r,R) + math.pow(2,R))/2
    else:
        mid = (r + 3*R)/4    

    print(mid)
    if mid > 1:
        
        l_best = 0.0001
        u_best = 0.0001
        diff = float("inf")
        for i in range(100000):
            u = random.uniform(0.0001, 100)
            l = random.uniform(0.0001, u)
            if abs(g(l,u)-mid) < diff:
                diff = abs(g(l,u)-mid)
                l_best = l
                u_best = u
        print(l_best, u_best)
        print("l_best: ", l_best, "u_best: ", u_best, "mid: ", mid, "g(l_best,u_best) ", g(l_best, u_best), "diff: ", diff, "and ", math.pow(2,R))
        return l_best, u_best
    else:
        return 5000, 500