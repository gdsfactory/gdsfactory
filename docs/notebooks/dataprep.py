# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Dataprep
#
# When building a reticle sometimes you want to do boolean operations
#

# + vscode={"languageId": "python"}
from gdsfactory.generic_tech.layer_map import LAYER as l
import gdsfactory.dataprep as dp
import gdsfactory as gf

gf.config.rich_output()
PDK = gf.generic_tech.get_generic_pdk()
PDK.activate()

# + vscode={"languageId": "python"}
c = gf.Component()

device = c << gf.components.coupler_ring()
floorplan = c << gf.components.bbox(device.bbox, layer=l.FLOORPLAN)
c.write_gds("src.gds")
c
# -

# ## Size
#
# You can copy/size layers

# + vscode={"languageId": "python"}
d = dp.Layout(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()  # copy layer
d.SLAB150 += 4  # size layer by 4 um
d.SLAB150 -= 2  # size layer by 2 um
c = d.write("dst.gds")
c
# -

# ## Booleans
#
# You can derive layers and do boolean operations.


# + vscode={"languageId": "python"}
d = dp.Layout(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()
d.SLAB150 += 3  # size layer by 3 um
d.SHALLOW_ETCH = d.SLAB150 - d.WG
c = d.write("dst.gds")
c
# -


# ## Fill
#
# You can add rectangular fill, using booleans to decide where to add it:

# + vscode={"languageId": "python"}
d = dp.Layout(filepath="src.gds", layermap=dict(l))

fill_region = d.FLOORPLAN - d.WG
fill_cell = d.get_fill(
    fill_region,
    size=[0.1, 0.1],
    spacing=[0.1, 0.1],
    fill_layers=[l.WG, l.M1],
    fill_name="test",
)
fill_cell
# -

# ## KLayout operations
#
# Any operation from Klayout Region can be called directly:

# + vscode={"languageId": "python"}
d = dp.Layout(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()
d.SLAB150.round_corners(1 * 1e3, 1 * 1e3, 100)  # round corners by 1um
c = d.write("dst.gds")
c
# -

# ## Parallel processing
#
# You can use dask for parallel processing


# + vscode={"languageId": "python"}
import dask
from IPython.display import HTML

dask.config.set(scheduler="threads")

c = gf.components.coupler_ring()
c.write_gds("src.gds")
c

# + vscode={"languageId": "python"}
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


# + vscode={"languageId": "python"}
# evaluation of the task graph is lazy
d.calculate()
c = d.write("dst.gds")
c
