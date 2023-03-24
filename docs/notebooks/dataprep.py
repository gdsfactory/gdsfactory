# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dataprep
#
# When building a reticle sometimes you want to do boolean operations
#

# %% tags=[]
from gdsfactory.generic_tech.layer_map import LAYER as l
import gdsfactory.dataprep as dp
import gdsfactory as gf

c = gf.components.coupler_ring()
c.write_gds("src.gds")
c
# %% [markdown]
# ## Size
#
# You can copy/size layers

# %%
d = dp.Layout(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()  # copy layer
d.SLAB150 += 4  # size layer by 4 um
d.SLAB150 -= 2  # size layer by 2 um
c = d.write("dst.gds")
c

# %% [markdown]
# ## Booleans
#
# You can derive layers and do boolean operations.


# %%
d = dp.Layout(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()
d.SLAB150 += 3  # size layer by 3 um
d.SHALLOW_ETCH = d.SLAB150 - d.WG
c = d.write("dst.gds")
c


# %% [markdown]
# ## Parallel processing
#
# You can use dask for parallel processing


# %%
import dask
from IPython.display import HTML

dask.config.set(scheduler="threads")

c = gf.components.coupler_ring()
c.write_gds("src.gds")
c

# %%
d = dp.Layout(filepath="src.gds", layermap=dict(l))
# you can do a bunch of derivations just to get a more interesting task graph
d.SLAB150 = d.WG + 3
d.SHALLOW_ETCH = d.SLAB150 - d.WG
d.DEEP_ETCH = d.WG + 2
d.M1 = d.DEEP_ETCH + 1
d.M2 = d.DEEP_ETCH - d.SHALLOW_ETCH

# visualize the taskgraph and save as 'tasks.html'
d.visualize("tasks")
HTML(filename="tasks.html")


# %%
# evaluation of the task graph is lazy
d.calculate()
c = d.write("dst.gds")
c

# %%
