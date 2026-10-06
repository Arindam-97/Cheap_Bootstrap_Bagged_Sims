import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from All_Helpers.Target_Helpers import *

ks = [2, 3, 5, 10, 30, 50, 100, 300, 500]

def exp_dist(size, rng):
    return rng.exponential(scale=1.0, size=size)

def sample_mean(x):
    return np.mean(x)

dist = exp_dist
Hk = sample_mean


gen_targets(ks, Hk, dist, "targets.pkl", reps=12800000, reps_small=20000, seed=1)
