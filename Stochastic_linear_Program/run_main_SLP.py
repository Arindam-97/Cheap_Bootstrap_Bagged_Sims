import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from All_Helpers.simul_helpers import *
from All_Helpers.Target_Helpers import *

import numpy as np
import pandas as pd
import pickle
import pickle
import concurrent.futures
import os
import random
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Set parameters for the simulation.")
parser.add_argument('--name', type=str, required=True, help="Name of the simulation")
# Parse the command-line arguments
args = parser.parse_args()

input_set = args.name




# Simulation Parameters

def normal_dist(size, rng):
    return rng.normal(size=size)

def LP_min(x):
  mu = np.mean(x, axis = 0)
  return 3*mu - np.abs(2*mu + 0.05)


dist = normal_dist
Hk = LP_min

TARGETS_FILE = "targets_SLP_norm.pkl"

# fixed inputs
reps = 800 #How many times each experiment is repeated. Parallelized over small rep batches
reps_small = 20
alpha = 0.05
seed = 1
n_count = 8 # number of Ns to evaluate at.

n_max = 5000 
D_mults = [0.01, 0.1, 0.5, 0.75, 1.0, 1.1] # Will run for D = D_mult*n


# input = [k, B, stat, name]
sim_params = {
    "simulation_1":  [50,   1, "U", None],
    "simulation_2":  [50,   1, "V", None],
    "simulation_3":  [50,   2, "U", None],
    "simulation_4":  [50,   2, "V", None],
    "simulation_5":  [50,   5, "U", None],
    "simulation_6":  [50,   5, "V", None],

    "simulation_7":  [100,  1, "U", None],
    "simulation_8":  [100,  1, "V", None],
    "simulation_9":  [100,  2, "U", None],
    "simulation_10": [100,  2, "V", None],
    "simulation_11": [100,  5, "U", None],
    "simulation_12": [100,  5, "V", None],

    "simulation_13": [300,  1, "U", None],
    "simulation_14": [300,  1, "V", None],
    "simulation_15": [300,  2, "U", None],
    "simulation_16": [300,  2, "V", None],
    "simulation_17": [300,  5, "U", None],
    "simulation_18": [300,  5, "V", None],

    "simulation_19": [500,  1, "U", None],
    "simulation_20": [500,  1, "V", None],
    "simulation_21": [500,  2, "U", None],
    "simulation_22": [500,  2, "V", None],
    "simulation_23": [500,  5, "U", None],
    "simulation_24": [500,  5, "V", None],

    "simulation_25": [1000, 1, "U", None],
    "simulation_26": [1000, 1, "V", None],
    "simulation_27": [1000, 2, "U", None],
    "simulation_28": [1000, 2, "V", None],
    "simulation_29": [1000, 5, "U", None],
    "simulation_30": [1000, 5, "V", None],

    "simulation_31": [2000, 1, "U", None],
    "simulation_32": [2000, 1, "V", None],
    "simulation_33": [2000, 2, "U", None],
    "simulation_34": [2000, 2, "V", None],
    "simulation_35": [2000, 5, "U", None],
    "simulation_36": [2000, 5, "V", None],

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
