# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Materials
#
# - How can you register your own material refractive index info for a particular PDK?
# - How can you use the same material index when using different plugins (tidy3d, meep, lumerical ...)?
#
# You can define a material by name, real refractive index, complex refractive index (for loss) or by a function of wavelength.

# + tags=[]
import numpy as np

import gdsfactory as gf
import gdsfactory.simulation.gmeep as gm
import gdsfactory.simulation.gtidy3d as gt
from gdsfactory.components.taper import taper_sc_nc
from gdsfactory.pdk import Pdk

gf.config.rich_output()
PDK = gf.generic_tech.get_generic_pdk()
PDK.activate()

# + tags=[]
strip = gt.modes.Waveguide(
    wavelength=1.55,
    wg_width=0.5,
    wg_thickness=0.22,
    slab_thickness=0.0,
    ncore="si",
    nclad="sio2",
)
strip.plot_index()
# -

# ## Option 1: define material with a value

# + tags=[]
PDK.materials_index.update(sin=2)

# + tags=[]
strip = gt.modes.Waveguide(
    wavelength=1.55,
    wg_width=0.5,
    wg_thickness=0.22,
    slab_thickness=0.0,
    ncore="sin",
    nclad="sio2",
)
strip.plot_index()


# -

# ## Option 2: define material with a function


# + tags=[]
def sin(wav: float) -> float:
    w = [1.3, 1.5]
    n = [1.9, 2.1]
    p = np.polyfit(w, n, 1)
    return np.polyval(p, wav)


PDK.materials_index.update(sin=sin)

# + tags=[]
strip = gt.modes.Waveguide(
    wavelength=1.5,
    wg_width=0.5,
    wg_thickness=0.22,
    slab_thickness=0.0,
    ncore="sin",
    nclad="sio2",
)
strip.plot_index()

# + tags=[]
c = taper_sc_nc(length=10)
c

# + tags=[]
s = gt.get_simulation(c, plot_modes=True)
fig = gt.plot_simulation_xz(s)

# + tags=[]
c = taper_sc_nc(length=10)
gm.write_sparameters_meep(
    c, ymargin=3, run=False, wavelength_start=1.31, wavelength_stop=1.36
)


# -

# ## Register materials into a PDK
#
# You can register all `materials_index` functions into a PDK.


# + tags=[]
def sin(wav: float) -> float:
    w = [1.3, 1.5]
    n = [1.9, 2.1]
    p = np.polyfit(w, n, 1)
    return np.polyval(p, wav)


def si(wav: float) -> float:
    w = [1.3, 1.5]
    n = [3.45, 3.47]
    p = np.polyfit(w, n, 1)
    return np.polyval(p, wav)


materials_index = dict(sin=sin, si=si)

p = Pdk(name="fab_a", materials_index=materials_index)
