# # Ray Tune generic black-box optimiser
#
# [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) is a hyperparameter tuning library primarily developed for machine learning.
# However, it is suitable for generic black-box optimisation as well.
# For our purpose, it provides an interface for running simulations inside a given *search space* and optimising for a given *loss function* $\mathcal{L}$ using a given *algorithm*.
# It automatically manages checkpointing, logging and, importantly, parallel (or even distributed) computing.
#
# You can see installation instructions [here](https://docs.ray.io/en/latest/ray-overview/installation.html), but the regular pip install should work for most. Notably, ARM-based macOS support is experimental.
#
# ```console
# pip install "ray[tune,air]" hyperopt
# ```
#
# You can optimise a `mmi1x2` component for a transmission of $|S_{21}|^2 = 0.5$ (50% power) for a given wavelength using MEEP.

# +
from functools import partial

import numpy as np
import ray
import ray.air
import ray.air.session
from ray import tune

import gdsfactory as gf
import gdsfactory.simulation.gmeep as gm
from gdsfactory.config import PATH
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

tmp = PATH.optimiser
tmp.mkdir(exist_ok=True)
# -

# ## Loss function $\mathcal{L}$
#
# The loss function is very important and should be designed to be meaningful for your need.
#
# The easiest method to optimise for a specific value is to use $L_1$ or $L_2$ (MSE) loss. Different optimisation algorithms might prefer more or less aggressive behaviour close to target, so choose depending on that.
# $$
# \begin{align*}
#     L_1(x) &= |x_\text{target} - x|, \\
#     L_2(x) &= \left(x_\text{target} - x\right)^2
#     .
# \end{align*}
# $$
#


def loss_S21_L1(x, target):
    r"""Loss function. Returns :math:`$\sum_i L_1(x_i)$` and :math:`$x$` as a tuple"""
    return np.abs(target - x), x


# Let's select a target of $0.7$ for $S_{21}$

loss = partial(loss_S21_L1, target=0.5)

# ## Optimisation settings
#
# Here we specify the search space, the optimiser and its settings.
#
# <div class="alert alert-block alert-info">
#     <b>Note</b> Choosing a new optimiser often requires you to install a separate package, see <a href="https://docs.ray.io/en/latest/tune/api_docs/suggestion.html">Ray Tune → Search Algorithms</a> for details. Here one needs to install <a href="http://hyperopt.github.io/hyperopt/">Hyperopt</a>.
# </div>

# +
search_config = {
    "length_mmi": tune.uniform(0.05, 2),
    "width_mmi": tune.uniform(0.05, 2),
}

# pylint: disable=wrong-import-position,ungrouped-imports
from ray.tune.search.hyperopt import HyperOptSearch

tune_config = tune.TuneConfig(
    metric="loss",
    mode="min",
    search_alg=HyperOptSearch(),
    max_concurrent_trials=2,  # simulations to run in parallel
    num_samples=-1,  # max iterations, can be -1 for infinite
    time_budget_s=60
    * 20,  # time after which optimisation is stopped. May be useful along with ``num_samples=-1``.
)
# -

# ## Implement a *trainable* function
#
# You need to implement a function which can be *trained* to be improved w.r.t. our $\mathcal{L}$.
# In other words, we create a function for a single training step, which generates, runs, and returns output $\mathcal{L}(\vec{x})$ from simulations for given parameters $\vec{x}$. This may require a bit more effort and some shell scripting to get right depending on your simulations.
#
# Here we demonstrate a trainable for S-parameter simulations. The `write_sparameters_meep` returns $\mathbf{S}$ as a function of $\lambda$ given in $\text{µm}$. From this, we select $S_{21}(\lambda)$ and try to optimise for $\min_\text{geometry} \sum_\lambda (S_{21}(\lambda) - \text{target})$. In other words, that the transmission from 1 to 2 would be as close to target as possible for the given wavelength (or range of wavelengths).
#


def trainable_simulations(config):
    """Training step, or `trainable`, function for Ray Tune to run simulations and return results."""

    # Component to optimise
    component = gf.components.mmi1x2(**config)

    # Simulate and get output
    dirpath = tmp / ray.air.session.get_trial_id()

    meep_params = dict(
        component=component,
        run=True,
        dirpath=dirpath,
        wavelength_start=1.5,
        # wavelength_stop=1.6,
        wavelength_points=1,
    )

    if use_mpi := True:  # change this to false if no MPI support
        s_params = gm.write_sparameters_meep_mpi(
            cores=2, **meep_params  # set this to be same as in `tune.Tuner`
        )
        s_params = np.load(s_params)  # parallel version returns filepath to npz instead
    else:
        s_params = gm.write_sparameters_meep(**meep_params)

    s_params_abs = np.abs(s_params["o2@0,o1@0"]) ** 2

    loss_x, x = loss(s_params_abs)
    if not np.isscalar(x):  # for many wavelengths, consider sum and mean
        loss_x, x = loss_x.sum(), x.mean()

    return {"loss": loss_x, "value": x}

    # ALTERNATIVE
    # For a shell-based solution to more software, subprocess.run is recommended roughly as below
    # interpreter = shutil.which('bash')
    # subprocess.run(
    #     [interpreter, '-c', './generated_simulation.sh'],
    #     cwd=dirpath,
    #     check=True,
    # )


# ## Run optimiser
# In the cell below, we gather all the code above to a [`tune.Tuner`](https://docs.ray.io/en/latest/tune/api_docs/execution.html#tuner) instance and start the optimisation by calling `tuner.fit()`.

# +
tuner = tune.Tuner(
    tune.with_resources(
        trainable_simulations, {"cpu": 2}
    ),  # maximum resources given to a worker, it also supports 'gpu'
    param_space=search_config,
    tune_config=tune_config,
    run_config=ray.air.RunConfig(
        local_dir=tmp / "ray_results",
        checkpoint_config=ray.air.CheckpointConfig(checkpoint_frequency=1),
        log_to_file=True,
        verbose=2,  # Intermediate results in Jupyter
    ),
)

# Previous simulations can be restored with, see https://docs.ray.io/en/latest/tune/tutorials/tune-stopping.html
# tuner = Tuner.restore(path=tmp / "ray_results/my_experiment")

results = tuner.fit()
# -

# The results can be seen and manipulated in DataFrame

df = results.get_dataframe()
df

# There are clearly many possible solutions, so making a [Pareto front](https://en.wikipedia.org/wiki/Pareto_front) plot w.r.t. some other parameter like overall size would make sense here.

best_params = results.get_best_result(metric="loss", mode="min").metrics
best_params["loss"], best_params["config"]
