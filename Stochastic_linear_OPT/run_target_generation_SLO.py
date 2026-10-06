import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from All_Helpers.Target_Helpers import *

ks = [1, 2, 5, 10, 30, 50, 100, 200, 500]



def simplex_dist(size, rng):
    """
    Returns samples of xi in R^10.
    First 5 coordinates ~ N(0,1)
    Last 5 coordinates  ~ N(0.1,1)

    Output shape: (*size, 10)
    """
    size = (size,) if np.isscalar(size) else tuple(size)

    mean = np.array([0.0]*5 + [0.1]*5)   # shape (10,)
    return rng.normal(loc=mean, scale=1.0, size=(*size, 10))


def simplex_SAA_value(x):
    """
    x: array of shape (k, 10)
    Returns the SAA optimal value:
        min_i average_j x[j,i]
    """
    sample_means = np.mean(x, axis=0)    # shape (10,)
    return float(np.min(sample_means))


dist = simplex_dist
Hk = simplex_SAA_value


gen_targets(ks, Hk, dist, "targets_SLO.pkl", reps=3200000, reps_small=10000, seed=1)
