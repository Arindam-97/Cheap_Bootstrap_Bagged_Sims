# Cheap Bootstrap for Bagged Estimators

Simulation code for studying cheap-bootstrap confidence intervals for finite-ensemble bagged estimators. Experiments vary the data size, subsample size, number of bags, and number of bootstrap replications, and report empirical coverage and average interval length.

## Experiments

| Directory | Base estimator | Simulation driver | Target-generation script |
|---|---|---|---|
| `Decision_Trees/` | Regression-tree prediction at a fixed test point | `run_main_DT.py` | `run_target_generation_DT.py` |
| `Stochastic_linear_OPT/` | Minimum of ten sample coordinate means (simplex SAA optimal value) | `run_main_SLO.py` | `run_target_generation_SLO.py` |
| `Stochastic_linear_Program/` | Scalar optimal-value example: `3 * mean(x) - abs(2 * mean(x) + 0.05)` | `run_main_SLP.py` | `run_target_generation_SLP.py` |
| `Simple_Mean/` | Sample mean of standard normal observations | `run_simple_mean.py` | `run_target_generation.py` |
| `Simple_Mean_EXP/` | Sample mean of unit-rate exponential observations | `run_simple_mean.py` | `run_target_generation.py` |

`All_Helpers/simul_helpers.py` implements bagging, bootstrap intervals, and parallel coverage experiments. `All_Helpers/Target_Helpers.py` estimates population targets and their Monte Carlo standard errors. Experiment folders also contain exploratory notebooks, input lists, and Slurm scripts.

The tree example aggregates `DecisionTreeRegressor` predictions; it does not use scikit-learn's `RandomForestRegressor`.

## Method

- `n`: number of observations in the original dataset.
- `k`: number of observations in each bag.
- `D`: number of bags/base estimators being averaged.
- `B`: number of bootstrap replications.
- `stat="U"`: sample each bag without replacement; `stat="V"`: sample with replacement.

Each bootstrap replication resamples `n` observations with replacement and independently constructs a new ensemble of `D` bags. If the original estimate is `u_hat` and the bootstrap estimates are `u_star`, the interval is

```text
S² = mean((u_star - u_hat)²)
CI = u_hat ± t_quantile(1 - alpha/2, df=B) * sqrt(S²)
```

Coverage is evaluated against the subsample-size-dependent target `E[H_k(X_1, ..., X_k)]`, estimated separately by Monte Carlo. For trees, this is the expected base-tree prediction, rather than the true regression function. For optimization, it is the expected subsample SAA optimal value, rather than the population optimization value.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy pandas scikit-learn
```

Install Jupyter separately if using the notebooks. Dependency versions are not currently pinned. The scripts are configured for Linux/Slurm execution; they lack `if __name__ == "__main__":` guards and need adaptation for multiprocessing with the spawn start method, including typical macOS/Windows setups.

## Run an experiment

Run commands from the experiment directory: target and output paths are relative to the current working directory. Generate targets first, then run a configuration selected by `--name`.

### Regression trees

```bash
cd Decision_Trees
mkdir -p pickles logs
python run_target_generation_DT.py
python run_main_DT.py --name simulation_1
```

The current target generator includes `k = [5, 20, 50, 100, 150, 200]`. `simulation_1` uses `k=50`, `B=1`, and U-type bagging. Before running other configurations, ensure their `k` values are included in the target generator's `ks` list and regenerate targets as needed.

### Stochastic optimization

```bash
cd Stochastic_linear_OPT
mkdir -p pickles logs
python run_target_generation_SLO.py
python run_main_SLO.py --name simulation_1
```

Here, `simulation_1` uses `k=10`, `B=1`, and U-type bagging. For the other experiment folders, use the corresponding scripts in the table above.

Target-generation defaults range from hundreds of thousands to millions of replications per subsample size. Reduce `reps` and `reps_small` in the target-generation script for a preliminary run. Also reduce the simulation driver's `reps`, `n_max`, and grid settings before a quick trial. Smaller target runs increase the uncertainty in the coverage reference.

## Configuration and parallel execution

Configurations are defined in each driver's `sim_params` dictionary. Other settings are edited directly in the driver:

| Setting | Meaning |
|---|---|
| `reps` | Requested number of repeated datasets per grid point |
| `reps_small` | Number of datasets per parallel batch |
| `alpha` | Interval significance level; default `0.05` |
| `seed` | Random seed |
| `n_count`, `n_max` | Grid of data sizes, from `k` to `n_max` |
| `D_mults` | Ensemble-size multipliers: `D = max(round(D_mult * n), 1)` |

Choose `reps` divisible by `reps_small`: the helpers run `reps // reps_small` complete batches and omit any remainder.

Coverage simulations use at most `SLURM_CPUS_PER_TASK` workers, defaulting to one when unset. Target generation uses `multiprocessing.cpu_count()` workers, independently of this setting.

For Slurm, run from the experiment folder after creating `logs/` and `pickles/`:

```bash
sbatch jobs_1.slurm
```

Adapt the account, environment, resource requests, and array range for your cluster. Each array task reads a configuration name from the corresponding `inputs_*.txt` file. Check the invoked Python script before submitting: some scripts contain copied settings; for example, `Decision_Trees/jobs_test.slurm` currently invokes `run_main_SLP.py`.

## Outputs

- Target files store `[targets, standard_errors]`, with both dictionaries indexed by `k`.
- Each simulation saves a pandas DataFrame to `pickles/<simulation_name>_<configuration>.pkl`.
- Columns are `k`, `stat`, `n`, `d` (the number of bags), `D_mult`, `B`, `coverage`, `average length`, `n_jobs`, and `simul_size` (the actual number of repeated datasets).

To inspect results from an experiment folder:

```python
import pandas as pd
from pathlib import Path

results = pd.concat(
    [pd.read_pickle(path) for path in sorted(Path("pickles").glob("*.pkl"))],
    ignore_index=True,
)
print(results[["k", "n", "d", "B", "stat", "coverage", "average length"]])
```

Generated results and logs are excluded by `.gitignore`. The current simulation drivers evaluate the cheap-bootstrap interval; benchmark methods and runtime measurements are not yet implemented in these drivers.
