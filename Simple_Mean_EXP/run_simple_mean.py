import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from All_Helpers.simul_helpers import *
from All_Helpers.Target_Helpers import *

import numpy as np
import pickle
import numpy as np
import pickle
import os
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Set parameters for the simulation.")
parser.add_argument('--name', type=str, required=True, help="Name of the simulation")
# Parse the command-line arguments
args = parser.parse_args()

input_set = args.name




# Simulation Parameters

def exp_dist(size, rng):
    return rng.exponential(scale=1.0, size=size)

def sample_mean(x):
    return np.mean(x)


dist = exp_dist
Hk = sample_mean

TARGETS_FILE = "targets.pkl"

# fixed inputs
reps = 800 #How many times each experiment is repeated. Parallelized over small rep batches
reps_small = 20

n_max = 5000 
n_count = 8 # number of Ns to evaluate at.

# simulation will run from n = k to n = n_max, with n_count many values of n
# example: k = 5, n_max = 15, n_count = 3 will run simulation for Ns = [5,10,15]


D_mults = [0.01,0.02,0.1,0.75, 1.0, 1.1] # Will run for D = D_mult*n

alpha = 0.05
seed = 1






# input = [k, B, stat, name]
sim_params = {
    "simulation_1":  [5,   1, "U", f"k5_B1_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_2":  [5,   1, "V", f"k5_B1_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_3":  [5,   2, "U", f"k5_B2_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_4":  [5,   2, "V", f"k5_B2_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_5":  [5,   5, "U", f"k5_B5_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_6":  [5,   5, "V", f"k5_B5_V_{dist.__name__}_{Hk.__name__}"],

    "simulation_7":  [10,  1, "U", f"k10_B1_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_8":  [10,  1, "V", f"k10_B1_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_9":  [10,  2, "U", f"k10_B2_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_10": [10,  2, "V", f"k10_B2_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_11": [10,  5, "U", f"k10_B5_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_12": [10,  5, "V", f"k10_B5_V_{dist.__name__}_{Hk.__name__}"],

    "simulation_13": [50,  1, "U", f"k50_B1_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_14": [50,  1, "V", f"k50_B1_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_15": [50,  2, "U", f"k50_B2_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_16": [50,  2, "V", f"k50_B2_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_17": [50,  5, "U", f"k50_B5_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_18": [50,  5, "V", f"k50_B5_V_{dist.__name__}_{Hk.__name__}"],

    "simulation_19": [100, 1, "U", f"k100_B1_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_20": [100, 1, "V", f"k100_B1_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_21": [100, 2, "U", f"k100_B2_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_22": [100, 2, "V", f"k100_B2_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_23": [100, 5, "U", f"k100_B5_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_24": [100, 5, "V", f"k100_B5_V_{dist.__name__}_{Hk.__name__}"],

    "simulation_25": [300, 1, "U", f"k300_B1_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_26": [300, 1, "V", f"k300_B1_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_27": [300, 2, "U", f"k300_B2_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_28": [300, 2, "V", f"k300_B2_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_29": [300, 5, "U", f"k300_B5_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_30": [300, 5, "V", f"k300_B5_V_{dist.__name__}_{Hk.__name__}"],

    "simulation_31": [500, 1, "U", f"k500_B1_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_32": [500, 1, "V", f"k500_B1_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_33": [500, 2, "U", f"k500_B2_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_34": [500, 2, "V", f"k500_B2_V_{dist.__name__}_{Hk.__name__}"],
    "simulation_35": [500, 5, "U", f"k500_B5_U_{dist.__name__}_{Hk.__name__}"],
    "simulation_36": [500, 5, "V", f"k500_B5_V_{dist.__name__}_{Hk.__name__}"],
}



k, B, stat, name = sim_params[input_set]

with open(TARGETS_FILE, "rb") as f:
	targets = pickle.load(f)[0]

target = targets[k]

out = run_grid_experiment(target, k, B, dist, Hk, stat, reps,reps_small, alpha, seed, n_count, n_max, D_mults)

with open(f"pickles/{input_set}_{name}.pkl", "wb") as f:
    pickle.dump(out, f)
