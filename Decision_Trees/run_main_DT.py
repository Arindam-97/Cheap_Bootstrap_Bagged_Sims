import os
import sys
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from All_Helpers.simul_helpers import *
from All_Helpers.Target_Helpers import *

import numpy as np
import pandas as pd
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




def tree_dist(size, rng, p=5, noise_sd=0.5):
    """
    Returns samples of xi = (X, Y).

    X in R^5 with independent Unif(0,1) coordinates.
    Y = sin(2*pi*X1) + X2^2 + 0.5*X3 - X4 + noise.

    Output shape: (*size, 6)
    Last column is Y.
    """
    size = (size,) if np.isscalar(size) else tuple(size)

    X = rng.uniform(0.0, 1.0, size=(*size, p))

    mean_y = (
        np.sin(2 * np.pi * X[..., 0])
        + X[..., 1] ** 2
        + 0.5 * X[..., 2]
        - X[..., 3]
    )

    eps = rng.normal(loc=0.0, scale=noise_sd, size=size)
    Y = mean_y + eps

    return np.concatenate([X, Y[..., None]], axis=-1)


class DecisionTreeValue:
    """
    Pickleable H_k for multiprocessing.

    Input x has shape (k, 6):
        first 5 columns = covariates X
        last column = response Y

    Output:
        decision tree prediction at fixed test point x0.
    """

    def __init__(self, x0=None, max_depth=4):
        if x0 is None:
            x0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        self.x0 = np.asarray(x0).reshape(1, -1)
        self.max_depth = max_depth
        self.name = "Decision_Tree"

    def __call__(self, x):
        k = x.shape[0]

        X = x[:, :-1]
        Y = x[:, -1]

        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=max(2, int(np.ceil(k / 10))),
            random_state=123,
        )

        tree.fit(X, Y)

        return float(tree.predict(self.x0)[0])


dist = tree_dist
Hk = DecisionTreeValue(max_depth=4)

Hk.__name__ = "Decision_Tree"
dist.__name__ = "tree_dist_1"




TARGETS_FILE = "targets_DT.pkl"

# fixed inputs
reps = 600 #How many times each experiment is repeated. Parallelized over small rep batches
reps_small = 60
alpha = 0.05
seed = 1
n_count = 2 # number of n's to evaluate at.

n_max = 5000 
D_mults = [0.01, 0.1, 0.5, 1, 1.1] # Will run for D = D_mult*n


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

    "simulation_31": [750, 1, "U", None],
    "simulation_32": [750, 1, "V", None],
    "simulation_33": [750, 2, "U", None],
    "simulation_34": [750, 2, "V", None],
    "simulation_35": [750, 5, "U", None],
    "simulation_36": [750, 5, "V", None],

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

    "simulation_49": [5, 5, "U", None],
    "simulation_50": [20, 5, "U", None],
    "simulation_51": [50, 5, "U", None],
    "simulation_52": [100, 5, "U", None],
    "simulation_53": [150, 5, "U", None],
    "simulation_54": [200, 5, "U", None],
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
