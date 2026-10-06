import numpy as np
from scipy.stats import t
from multiprocessing import Pool
import pandas as pd
import os


def draw_subsamples(n, k, D, rng, replace):
    if replace:
        return rng.choice(n, size=(D, k), replace=True)

    out = np.empty((D, k), dtype=int)
    for d in range(D):
        out[d] = rng.choice(n, size=k, replace=False)
    return out


def bagged_estimate(x, Hk, subsamples):
    return np.mean([Hk(x[idx]) for idx in subsamples])


def bootstrap_bagged_estimates(x, Hk, k, D, B, rng, replace):
    n = len(x)
    vals = np.empty(B)
    for b in range(B):
        xb = x[rng.choice(n, size=n, replace=True)]
        subsamples_b = draw_subsamples(n, k, D, rng, replace=replace)
        vals[b] = bagged_estimate(xb, Hk, subsamples_b)
    return vals


def cheap_bootstrap_interval(x, Hk, k, D, B, rng, alpha=0.05, stat="U"):
    replace = (stat == "V")

    subsamples = draw_subsamples(len(x), k, D, rng, replace=replace)
    u_hat = bagged_estimate(x, Hk, subsamples)

    u_star = bootstrap_bagged_estimates(x, Hk, k, D, B, rng, replace=replace)

    S2 = np.mean((u_star - u_hat) ** 2)
    S = np.sqrt(S2)

    q = t.ppf(1 - alpha / 2, df=B)
    ci = (u_hat - q * S, u_hat + q * S)

    return {
        "estimate": u_hat,
        "bootstrap_estimates": u_star,
        "S2": S2,
        "S": S,
        "ci": ci,
    }


def data_generation(n, reps_small, dist, rng):
    return dist(size=(n, reps_small), rng = rng)


def _run_one_batch(args):
    batch_id, n, reps_small, dist, Hk, k, D, B, stat, target, alpha, seed = args

    rng = np.random.default_rng(seed)
    X = data_generation(n, reps_small, dist, rng)

    lengths = np.empty(reps_small)
    covered = np.empty(reps_small, dtype=int)

    for j in range(reps_small):
        out = cheap_bootstrap_interval(
            x=X[:, j],
            Hk=Hk,
            k=k,
            D=D,
            B=B,
            stat=stat,
            rng=rng,
            alpha=alpha,
        )

        L, U = out["ci"]
        lengths[j] = U - L
        covered[j] = (L <= target <= U)

    return lengths, covered


def run_experiment_batched_parallel(
    n, reps, reps_small, dist, Hk, k, D, B, stat, target,
    alpha=0.05, seed=None
):
    n_batches = reps // reps_small
    n_jobs = min(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), n_batches)

    seed_seq = np.random.SeedSequence(seed)
    batch_seeds = seed_seq.spawn(n_batches)

    args = [
        (b, n, reps_small, dist, Hk, k, D, B, stat, target, alpha, batch_seeds[b])
        for b in range(n_batches)
    ]

    with Pool(processes=n_jobs) as pool:
        results = pool.map(_run_one_batch, args)

    lengths = np.concatenate([r[0] for r in results])
    covered = np.concatenate([r[1] for r in results])

    return {
        "mean_interval_length": lengths.mean(),
        "mean_coverage_probability": covered.mean(),
        "n_batches": n_batches,
        "n_jobs": n_jobs,
	"reps": len(lengths)
    }





def run_grid_experiment(target, k, B, dist, Hk, stat, reps,reps_small, alpha = 0.05, seed = 1, n_count=10, n_max = 5000, D_mults = [0.75, 1.0, 1.1]):
    Ns = np.unique(np.round(np.linspace(k, n_max, n_count)).astype(int))
    rows = []

    for n in Ns:
        print("################")
        print("n = ",n)
        for D_mult in D_mults:
            D = max(int(round(D_mult * n)),1)
            print("D = ",D)
            res = run_experiment_batched_parallel(
                n=n,
                reps=reps,
                reps_small=reps_small,
                dist=dist,
                Hk=Hk,
                k=k,
                D=D,
                B=B,
                stat=stat,
                target=target,
                alpha=alpha,
                seed=seed
            )

            rows.append({
                "k": k,
                "stat": stat,
                "n": n,
                "d": D,
                "D_mult": D_mult,
                "B": B,
                "coverage": res["mean_coverage_probability"],
                "average length": res["mean_interval_length"],
                "n_jobs": res["n_jobs"],
		"simul_size": res["reps"]
            })

    return pd.DataFrame(rows)
