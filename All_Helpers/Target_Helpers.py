import os
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count


def _target_worker(args):
    k, H_k, dist, reps_small, seed = args
    rng = np.random.default_rng(seed)
    vals = np.empty(reps_small)
    for j in range(reps_small):
        vals[j] = H_k(dist(size=k, rng=rng))
    return vals


def target_mapper(k, H_k, dist, reps=100000, reps_small=1000, seed=1):
    n_batches = reps // reps_small
    seeds = np.random.SeedSequence(seed).spawn(n_batches)

    args = [(k, H_k, dist, reps_small, seeds[b]) for b in range(n_batches)]

    with Pool(cpu_count()) as pool:
        vals = np.concatenate(pool.map(_target_worker, args))
    return vals.mean(), vals.std()/np.sqrt(len(vals))


def gen_targets(ks, H_k, dist, filename, reps=200000, reps_small=1000, seed=1):
    targets = {}
    sd = {}
    k_s = np.random.SeedSequence(seed).spawn(len(ks))
    k_seeds = [int(s.generate_state(1)[0]) for s in k_s]
    for i, k in enumerate(ks):
        print(f"Computing target for {k}")
        targets[k], sd[k] = target_mapper(k, H_k, dist, reps, reps_small, k_seeds[i])
    res = [targets,sd]
    with open(filename, "wb") as f:
        pickle.dump(res, f)
