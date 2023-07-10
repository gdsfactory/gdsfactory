# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Particle Swarm optimization
#
# Example of particle swarm implementation based on PySwarms.
# This code demonstrates how to use the Particle Swarm Optimization (PSO) algorithm
# from the PySwarms library to optimize a function.
# The objective function used in this example is taken from the Ray Tune generic black-box optimizer.

# %% [markdown]
# PySwarms: https://pyswarms.readthedocs.io/en/latest/
# To install PySwarms, use the following command from console:
#
# ```console
# pip install pyswarms
# ```
#
# You can optimise a `mmi1x2` component for a transmission of $|S_{21}|^2 = 0.5$ (50% power) for a given wavelength using MEEP.

# %%
from functools import partial
import numpy as np

import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

import gdsfactory as gf
import gdsfactory.simulation.gmeep as gm
from gdsfactory.config import PATH
from gdsfactory.generic_tech import get_generic_pdk

# Set up GDSFactory and PDK
gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# Create a working directory for the PSO optimization
wrk_dir = PATH.cwd / "extra"
wrk_dir.mkdir(exist_ok=True)


# ## Define the loss function used in the PSO optimization
def loss_S21_L1(x, target):
    r"""Loss function. Returns :math:`$\sum_i L_1(x_i)$` and :math:`$x$` as a tuple"""
    return np.abs(target - x), x


# ## Define the trainable function for the PSO optimization
def trainable_simulations(x, loss=lambda x: x):
    """Training step, or `trainable`, function for Ray Tune to run simulations and return results."""
    loss_arr = []
    use_mpi = False

    for xi in x:
        # Component to optimize
        component = gf.components.mmi1x2(length_mmi=xi[0], width_mmi=xi[1])

        # Simulate and get output
        meep_params = dict(
            component=component,
            run=True,
            dirpath=wrk_dir,
            wavelength_start=1.5,
            wavelength_points=1,
            is_3d=False,
        )
        if use_mpi:  # change this to false if no MPI support
            s_params = gm.write_sparameters_meep_mpi(
                cores=2, **meep_params  # set this to be the same as in `tune.Tuner`
            )
            s_params = np.load(
                s_params
            )  # parallel version returns the filepath to npz instead
        else:
            s_params = gm.write_sparameters_meep(**meep_params)

        s_params_abs = np.abs(s_params["o2@0,o1@0"]) ** 2

        loss_x, x = loss(s_params_abs)
        if not np.isscalar(x):  # for many wavelengths, consider sum and mean
            loss_x, x = -loss_x.sum(), x.mean()

        loss_arr.append(loss_x)

    return np.asarray(loss_arr)


# Define the target value for the loss function
loss = partial(loss_S21_L1, target=0.5)
func = partial(trainable_simulations, loss=loss)

# Create bounds for the optimization
max_bound = np.array([0.05, 0.05])
min_bound = np.array([2, 2])
bounds = (min_bound, max_bound)

# Set options for the PSO optimizer
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

# Create an instance of the PSO optimizer
optimizer = ps.single.GlobalBestPSO(
    n_particles=10, dimensions=2, options=options, bounds=bounds
)


# %%
# Perform the optimization. We run only comment these lines for demo purposes
# cost, pos = optimizer.optimize(func, iters=100)
# plot_cost_history(cost_history=optimizer.cost_history)
# plt.show()
