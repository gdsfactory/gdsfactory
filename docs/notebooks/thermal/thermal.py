# ---
# jupyter:
#   jupytext:
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

# # Thermal
#
# gdsfactory has an FEM [femwell](https://gdsfactory.github.io/gdsfactory/notebooks/fem/01_mode_solving.html) plugin that you can use for thermal simulations.
# You can simulate directly the component layout and include important effects such as metal dummy fill.


# +
import gmsh
import gdsfactory as gf
from gdsfactory.simulation.gmsh.mesh import create_physical_mesh
from gdsfactory.simulation.thermal import solve_thermal
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.technology import LayerStack, LayerLevel
import meshio
from skfem.io import from_meshio

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

gf.generic_tech.LAYER_STACK.layers["heater"].thickness = 0.13
gf.generic_tech.LAYER_STACK.layers["heater"].zmin = 2.2

heater = gf.components.straight_heater_metal(length=50, heater_width=2)
heater
# -

print(gf.generic_tech.LAYER_STACK.layers.keys())

filtered_layerstack = LayerStack(
    layers={
        k: gf.pdk.get_layer_stack().layers[k]
        for k in ("slab90", "core", "via_contact", "heater")
    }
)

# +
filename = "mesh"


def mesh_with_physicals(mesh, filename):
    mesh_from_file = meshio.read(f"{filename}.msh")
    return create_physical_mesh(mesh_from_file, "triangle", prune_z=True)


# -

mesh = heater.to_gmsh(
    type="uz",
    xsection_bounds=[(4, -4), (4, 4)],
    layer_stack=filtered_layerstack,
    filename=f"{filename}.msh",
)
mesh = mesh_with_physicals(mesh, filename)
mesh = from_meshio(mesh)
mesh.draw().plot()

# FIXME!
#
# ```python
#
# solve_thermal(
#     mesh_filename="mesh.msh",
#     thermal_conductivity={"heater": 28, "oxide": 1.38, "core": 148},
#     specific_conductivity={"heater": 2.3e6},
#     thermal_diffusivity={
#         "heater": 28 / 598 / 5240,
#         "oxide": 1.38 / 709 / 2203,
#         "core": 148 / 711 / 2330,
#     },
#     # specific_heat={"(47, 0)_0": 598, 'oxide': 709, '(1, 0)': 711},
#     # density={"(47, 0)_0": 5240, 'oxide': 2203, '(1, 0)': 2330},
#     currents={"heater": 0.007},
# )
# ```

# Example based on [femwell](https://helgegehring.github.io/femwell/index.html)

# +
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon
from skfem import Basis, ElementTriP0
from skfem.io import from_meshio
from tqdm import tqdm

from femwell.mesh import mesh_from_OrderedDict
from femwell.mode_solver import compute_modes, plot_mode
from femwell.thermal import solve_thermal

# +
w_sim = 8 * 2
h_clad = 2.8
h_box = 1
w_core = 0.5
h_core = 0.22
offset_heater = 2.2
h_heater = 0.14
w_heater = 2

polygons = OrderedDict(
    bottom=LineString(
        [(-w_sim / 2, -h_core / 2 - h_box), (w_sim / 2, -h_core / 2 - h_box)]
    ),
    core=Polygon(
        [
            (-w_core / 2, -h_core / 2),
            (-w_core / 2, h_core / 2),
            (w_core / 2, h_core / 2),
            (w_core / 2, -h_core / 2),
        ]
    ),
    heater=Polygon(
        [
            (-w_heater / 2, -h_heater / 2 + offset_heater),
            (-w_heater / 2, h_heater / 2 + offset_heater),
            (w_heater / 2, h_heater / 2 + offset_heater),
            (w_heater / 2, -h_heater / 2 + offset_heater),
        ]
    ),
    clad=Polygon(
        [
            (-w_sim / 2, -h_core / 2),
            (-w_sim / 2, -h_core / 2 + h_clad),
            (w_sim / 2, -h_core / 2 + h_clad),
            (w_sim / 2, -h_core / 2),
        ]
    ),
    box=Polygon(
        [
            (-w_sim / 2, -h_core / 2),
            (-w_sim / 2, -h_core / 2 - h_box),
            (w_sim / 2, -h_core / 2 - h_box),
            (w_sim / 2, -h_core / 2),
        ]
    ),
)

resolutions = dict(
    core={"resolution": 0.04, "distance": 1},
    clad={"resolution": 0.6, "distance": 1},
    box={"resolution": 0.6, "distance": 1},
    heater={"resolution": 0.1, "distance": 1},
)

mesh = from_meshio(
    mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=0.6)
)
mesh.draw().show()
