# # EME with MEOW
#
# Some components are more efficiently modeled with Eigenmode Expansion.
#
# Gdsfactory provides a plugin for MEOW to efficiently extract component S-parameters through EME.
#
# Currently the component needs to specifically have a single "o1" port facing west, and a single "o2" port facing east, like this taper:

# +
import gdsfactory as gf
import matplotlib.pyplot as plt
import meow as mw
import numpy as np
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.simulation.eme import MEOW

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

c = gf.components.taper_cross_section_sine()
c.plot()
# -

# You also need to explicitly provide a LayerStack to define cross-sections, for instance the generic one:

# +
layerstack = gf.generic_tech.LAYER_STACK

filtered_layerstack = gf.technology.LayerStack(
    layers={
        k: layerstack.layers[k]
        for k in (
            "slab90",
            "core",
            "box",
            "clad",
        )
    }
)
# -

# Since you need to make sure that your entire LayerStack has e.g. material information for all present layers, it is safer to only keep the layers that you need for your simulation:

# The EME simulator can be instantiated with only these two elements, alongside parameters:

eme = MEOW(component=c, layerstack=filtered_layerstack, wavelength=1.55, overwrite=True)

# Plotting functions allow you to check your simulation:

eme.plot_structure()

# The cross-section themselves:

eme.plot_cross_section(xs_num=0)

eme.plot_cross_section(xs_num=-1)

# And the modes (after calculating them):

eme.plot_mode(xs_num=0, mode_num=0)

eme.plot_mode(xs_num=-1, mode_num=0)

# The S-parameters can be calculated, and are returned in the same format as for the FDTD solvers (the original MEOW S-parameter results S and port_names are saved as attributes):

sp = eme.compute_sparameters()

print(np.abs(sp["o1@0,o2@0"]) ** 2)

print(eme.port_map)
eme.plot_s_params()

# As you can see most light stays on the fundamental TE mode

# ## Sweep EME length
#
# Lets sweep the length of the taper.

# +
layerstack = gf.generic_tech.LAYER_STACK

filtered_layerstack = gf.technology.LayerStack(
    layers={
        k: layerstack.layers[k]
        for k in (
            "slab90",
            "core",
            "box",
            "clad",
        )
    }
)

c = gf.components.taper(width1=0.5, width2=2, length=10.0)
c.plot()
# -

# Lets do a convergence tests on the `cell_length` parameter. This depends a lot on the structure.

# +
import matplotlib.pyplot as plt

trans = []
cells_lengths = [0.1, 0.25, 0.5, 0.75, 1]

for cell_length in cells_lengths:
    m = MEOW(
        component=c,
        layerstack=filtered_layerstack,
        wavelength=1.55,
        overwrite=True,
        spacing_y=-3,
        cell_length=cell_length,
    )
    sp = m.compute_sparameters()
    te0_trans = np.abs(sp["o1@0,o2@0"]) ** 2
    trans.append(te0_trans)

plt.plot(cells_lengths, trans, ".-")
plt.title("10um taper, resx = resy = 100, num_modes = 4")
plt.xlabel("Cell length (um)")
plt.ylabel("TE0 transmission")
# -

eme = MEOW(
    component=c,
    layerstack=filtered_layerstack,
    wavelength=1.55,
    overwrite=True,
    spacing_y=-3,
    cell_length=0.25,
)

eme.plot_cross_section(xs_num=0)

eme.plot_mode(xs_num=0, mode_num=0)

eme.plot_cross_section(xs_num=-1)

eme.plot_mode(xs_num=-1, mode_num=0)

sp = eme.compute_sparameters()

print(eme.port_map)
eme.plot_s_params()

T = np.abs(sp["o1@0,o2@0"]) ** 2
T

np.abs(sp["o1@0,o2@2"]) ** 2

lengths = np.array([1, 2, 3, 5, 10, 20])
T = np.zeros_like(lengths, dtype=float)

for length in lengths:
    c = gf.components.taper(width1=0.5, width2=2, length=length)
    c.plot()

for i, length in enumerate(lengths):
    print(f"{length=}")
    c = gf.components.taper(width1=0.5, width2=2, length=length)
    eme = MEOW(
        component=c,
        layerstack=filtered_layerstack,
        wavelength=1.55,
        overwrite=True,
        spacing_y=-3,
        cell_length=0.25,
    )
    sp = eme.compute_sparameters()
    T[i] = np.abs(sp["o1@0,o2@0"]) ** 2

plt.plot(lengths, T, marker="o")
plt.ylim(0.6, 1.0)
plt.title("Fundamental mode transmission")
plt.ylabel("Transmission")
plt.xlabel("taper length (um)")
plt.grid(True)
plt.show()

eme.plot_s_params()
