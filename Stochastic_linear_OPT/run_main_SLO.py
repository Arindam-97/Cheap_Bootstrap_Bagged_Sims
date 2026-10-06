import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from All_Helpers.simul_helpers import *
from All_Helpers.Target_Helpers import *

import numpy as np
import pickle
import pickle
import os
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Set parameters for the simulation.")
parser.add_argument('--name', type=str, required=True, help="Name of the simulation")
# Parse the command-line arguments
args = parser.parse_args()

input_set = args.name



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


TARGETS_FILE = "targets_SLO.pkl"

# fixed inputs
reps = 800 #How many times each experiment is repeated. Parallelized over small rep batches
reps_small = 20

n_max = 5000 
n_count = 8 # number of Ns to evaluate at.

# simulation will run from n = k to n = n_max, with n_count many values of n
# example: k = 5, n_max = 15, n_count = 3 will run simulation for Ns = [5,10,15]


D_mults = [0.01, 0.1, 0.5, 0.75, 1.0, 1.1] # Will run for D = D_mult*n

alpha = 0.05
seed = 1



# input = [k, B, stat, name]
sim_params = {
    "simulation_1":  [10,   1, "U", None],
    "simulation_2":  [10,   1, "V", None],
    "simulation_3":  [10,   2, "U", None],
    "simulation_4":  [10,   2, "V", None],
    "simulation_5":  [10,   5, "U", None],
    "simulation_6":  [10,   5, "V", None],

    "simulation_7":  [30,  1, "U", None],
    "simulation_8":  [30,  1, "V", None],
    "simulation_9":  [30,  2, "U", None],
    "simulation_10": [30,  2, "V", None],
    "simulation_11": [30,  5, "U", None],
    "simulation_12": [30,  5, "V", None],

    "simulation_13": [50,  1, "U", None],
    "simulation_14": [50,  1, "V", None],
    "simulation_15": [50,  2, "U", None],
    "simulation_16": [50,  2, "V", None],
    "simulation_17": [50,  5, "U", None],
    "simulation_18": [50,  5, "V", None],

    "simulation_19": [100,  1, "U", None],
    "simulation_20": [100,  1, "V", None],
    "simulation_21": [100,  2, "U", None],
    "simulation_22": [100,  2, "V", None],
    "simulation_23": [100,  5, "U", None],
    "simulation_24": [100,  5, "V", None],

    "simulation_25": [200, 1, "U", None],
    "simulation_26": [200, 1, "V", None],
    "simulation_27": [200, 2, "U", None],
    "simulation_28": [200, 2, "V", None],
    "simulation_29": [200, 5, "U", None],
    "simulation_30": [200, 5, "V", None],

    "simulation_31": [500, 1, "U", None],
    "simulation_32": [500, 1, "V", None],
    "simulation_33": [500, 2, "U", None],
    "simulation_34": [500, 2, "V", None],
    "simulation_35": [500, 5, "U", None],
    "simulation_36": [500, 5, "V", None],

    "simulation_37": [2, 1, "U", None],
    "simulation_38": [2, 1, "V", None],
    "simulation_39": [2, 2, "U", None],
    "simulation_40": [2, 2, "V", None],
    "simulation_41": [2, 5, "U", None],
    "simulation_42": [2, 5, "V", None],

    "simulation_43": [5, 1, "U", None],
    "simulation_44": [5, 1, "V", None],
    "simulation_45": [5, 2, "U", None],
    "simulation_46": [5, 2, "V", None],
    "simulation_47": [5, 5, "U", None],
    "simulation_48": [5, 5, "V", None],
}

for key, val in sim_params.items():
    k, B, stat = val[:3]
    val[3] = f"k{k}_B{B}_{stat}_{dist.__name__}_{Hk.__name__}"



k, B, stat, name = sim_params[input_set]

with open(TARGETS_FILE, "rb") as f:
	targets = pickle.load(f)[0]

target = targets[k]

out = run_grid_experiment(target, k, B, dist, Hk, stat, reps,reps_small, alpha, seed, n_count, n_max, D_mults)

with open(f"pickles/{input_set}_{name}.pkl", "wb") as f:
    pickle.dump(out, f)
