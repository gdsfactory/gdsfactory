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
# # Maskprep
#
# When building a reticle sometimes you want to do boolean operations
#

# %%
from gdsfactory.generic_tech.layer_map import LAYER as l
import gdsfactory.geometry.maskprep_flat as dp
import gdsfactory as gf

gf.config.rich_output()
PDK = gf.generic_tech.get_generic_pdk()
PDK.activate()

# %%
c = gf.Component()

device = c << gf.components.coupler_ring()
floorplan = c << gf.components.bbox(device.bbox, layer=l.FLOORPLAN)
c.write_gds("src.gds")
c

# %% [markdown]
# ## Size
#
# You can copy/size layers

# %%
d = dp.RegionCollection(filepath="src.gds", layermap=dict(l))
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
d = dp.RegionCollection(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()
d.SLAB150 += 3  # size layer by 3 um
d.SHALLOW_ETCH = d.SLAB150 - d.WG
c = d.write("dst.gds")
c


# %% [markdown]
# ## Fill
#
# You can add rectangular fill, using booleans to decide where to add it:

# %%
d = dp.RegionCollection(filepath="src.gds", layermap=dict(l))

fill_region = d.FLOORPLAN - d.WG
fill_cell = d.get_fill(
    fill_region,
    size=[0.1, 0.1],
    spacing=[0.1, 0.1],
    fill_layers=[l.WG, l.M1],
    fill_name="test",
)
fill_cell

# %% [markdown]
# ## KLayout operations
#
# Any operation from Klayout Region can be called directly:

# %%
d = dp.RegionCollection(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()
d.SLAB150.round_corners(1 * 1e3, 1 * 1e3, 100)  # round corners by 1um
c = d.write("dst.gds")
c

# %%
