import os
import sys
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from All_Helpers.Target_Helpers import *

#ks = [1, 2, 5, 10, 50, 100, 300, 500, 1000, 750]
ks = [5, 20, 50, 100, 150, 200]
#ks = [50, 100]

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



gen_targets(ks, Hk, dist, "targets_DT.pkl", reps=320000, reps_small=10000, seed=1)
