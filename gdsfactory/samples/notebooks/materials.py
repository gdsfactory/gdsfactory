# # Materials
#
# - How can you register your own material refractive index info for a particular PDK?
# - How can you use the same material index when using different plugins (tidy3d, meep, lumerical ...)?
#
# You can define a material by name, real refractive index, complex refractive index (for loss) or by a function of wavelength.

# +
import numpy as np

import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt
from gdsfactory.components.taper import taper_sc_nc
from gdsfactory.pdk import Pdk

gf.config.rich_output()
PDK = gf.generic_tech.get_generic_pdk()
PDK.activate()
# -

strip = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=0.5,
    core_thickness=0.22,
    slab_thickness=0.0,
    core_material="si",
    clad_material="sio2",
)
strip.plot_index()

# ## Option 1: define material with a value

PDK.materials_index.update(sin=2)

strip = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=0.5,
    core_thickness=0.22,
    slab_thickness=0.0,
    core_material="sin",
    clad_material="sio2",
)
strip.plot_index()


# ## Option 2: define material with a function
#


# +
def sin(wav: float) -> float:
    w = [1.3, 1.5]
    n = [1.9, 2.1]
    p = np.polyfit(w, n, 1)
    return np.polyval(p, wav)


PDK.materials_index.update(sin=sin)
# -

strip = gt.modes.Waveguide(
    wavelength=1.5,
    core_width=0.5,
    core_thickness=0.22,
    slab_thickness=0.0,
    core_material="sin",
    clad_material="sio2",
)
strip.plot_index()

c = taper_sc_nc(length=10)
c.plot()

s = gt.get_simulation(c, plot_modes=True)
fig = gt.plot_simulation_xz(s)


# ## Register materials into a PDK
#
# You can register all `materials_index` functions into a PDK.
#


# +
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
