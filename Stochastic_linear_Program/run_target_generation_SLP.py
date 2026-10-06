import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from All_Helpers.Target_Helpers import *

ks = [1, 2, 5, 10, 50, 100, 300, 500, 1000, 2000]

def normal_dist(size, rng):
    return rng.normal(size=size)

def SLP(x):
    mu = np.mean(x)
    return 3*mu - np.abs(2*mu+0.05)

dist = normal_dist
Hk = SLP


gen_targets(ks, Hk, dist, "targets_SLP_norm.pkl", reps=3200000, reps_small=10000, seed=1)
