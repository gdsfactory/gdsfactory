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

c = gf.c.coupler_ring(cross_section="strip")
c.write_gds("src.gds")
c
# %% [markdown]
# ## Size
#
# You can copy/size layers

# %%
d = dp.Layout(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()
d.SLAB150 += 4  # size layer by 4 um
d.SLAB150 -= 2  # size layer by 2 um

# %%
# d.trench = l. - l.WG  # boolean NOT
# d.trench += 2  # size layer by 2 um
# d.text = l.TEXT  # copy a layer
# d.text.remove()  # delete a layer
# del d.SLAB150  # delete a layer
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
