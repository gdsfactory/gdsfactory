# # DEVSIM TCAD simulator
#
# [DEVSIM](https://devsim.org/) is an open-source semiconductor device simulator. See [publication](https://joss.theoj.org/papers/10.21105/joss.03898).
#
# Some of its features include:
#
# * Sharfetter-Gummel discretization of the electron and hole continuity equations
# * DC, transient, small-signal AC, and noise solution algorithms
# * Solution of 1D, 2D, and 3D unstructured meshes
# * Advanced models for mobility and semiclassical approaches for quantum effects
#
#
# It allows scripting new models and derivatives thanks to its a symbolic model evaluation interface
#
# There is an active community over at the [DEVSIM forums](https://forum.devsim.org/).
#
# ## Meshing
#
# DEVSIM solves equations on unstructured meshes.
# It has a built-in 1D and 2D meshing interface, you can solve carrier distributions in waveguide cross-sections.
# It also interfaces with GMSH for arbitrary 2D and 3D meshes, which you can use for running semiconductor simulations with gdsfactory components.
#
# ![](https://i.imgur.com/hsuzB5K.png)
#
# ## Install DEVSIM
#
# To install DEVSIM you can run `pip install devsim` or `pip install gdsfactory[full]`.

# ## DC Drift-diffusion simulation
#

# You can setup the simulation by defining a strip waveguide cross-section.
# You can change waveguide geometry (core thickness, slab thickness, core width), doping configuration (dopant level, dopant positions), as well as hyperparameters like adaptive mesh resolution at all the interfaces.

# +
import numpy as np
import matplotlib.pyplot as plt
from gdsfactory.simulation.devsim import get_simulation_xsection
from gdsfactory.simulation.devsim.get_simulation_xsection import k_to_alpha
import gdsfactory as gf

gf.config.rich_output()
PDK = gf.get_generic_pdk()
PDK.activate()

# +
# %%capture

nm = 1e-9
c = get_simulation_xsection.PINWaveguide(
    core_width=500 * nm,
    core_thickness=220 * nm,
    slab_thickness=90 * nm,
)

# Initialize mesh and solver
c.ddsolver()
# -

# You can save the device to a tecplot file named `filename.dat` with `c.save_device(filename=filename.dat)`, and then open with [Paraview](https://www.paraview.org/).
#
# You can also plot the mesh in the Notebook with the `plot` method. By default it shows the geometry.
# You can also pass a string to `scalars` to plot a field as color over the mesh.
# For instance, acceptor concentration and donor concentration for the PN junction.
#
# `list_fields()` returns the header of the mesh, which lists all possible fields.

c.list_fields()

# Finite-element field information can be plotted using pyvista (note that lengths in DEVSIM are cm by default):

c.plot(scalars="NetDoping")

c.plot(scalars="Electrons", log_scale=True)

# ### Solve
#
# Using default DEVSIM silicon models, we iteratively solve for the self-consistent carrier distribution for 0.5V of applied forward voltage, iterating with 0.1V steps, and then visualize the electron concentration:

# %%capture
# Find a solution with 1V across the junction, ramping by 0.1V steps
c.ramp_voltage(Vfinal=0.5, Vstep=0.1)

c.plot(scalars="Electrons", log_scale=True)

# and similarly for reverse-bias:

# %%capture
c.ramp_voltage(Vfinal=-0.5, Vstep=-0.1)

c.plot(scalars="Electrons", log_scale=True)

# # mode solver interface
#
# The carrier distribution can be used to create a mode solver object with perturbed index, and to acquire the effective index as a function of applied voltage:

# +
# %%capture
voltages = [0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4]
ramp_rate = -0.1

n_dist = {}
neffs = {}

for ind, voltage in enumerate(voltages):
    Vinit = 0 if ind == 0 else voltages[ind - 1]
    c.ramp_voltage(Vfinal=voltage, Vstep=ramp_rate, Vinit=Vinit)
    waveguide = c.make_waveguide(wavelength=1.55)
    n_dist[voltage] = waveguide.index.values
    neffs[voltage] = waveguide.n_eff[0]

# +
voltage_list = sorted(neffs.items())
x, y = zip(*voltage_list)

plt.plot(x, np.real(y) - neffs[0])

plt.xlabel("Voltage (V)")
plt.ylabel(r"$\Delta n_{eff}$")

# +
voltage_list = sorted(neffs.items())
x, y = zip(*voltage_list)

plt.plot(x, -10 * np.log10(1 - k_to_alpha(np.imag(y), wavelength=1.55)))

plt.xlabel("Voltage (V)")
plt.ylabel(r"$\alpha (dB/cm)$")
# -

# We can compare the index distribution the same undoped waveguide:

c_undoped = c.make_waveguide(wavelength=1.55, perturb=False, precision="double")
c_undoped.compute_modes()
n_undoped = c_undoped.index.values

plt.imshow(
    np.log(np.abs(np.real(n_dist[0].T - n_undoped.T))),
    origin="lower",
    extent=[
        -c.xmargin - c.ppp_offset - c.core_width / 2,
        c.xmargin + c.npp_offset + c.core_width / 2,
        0,
        c.clad_thickness + c.box_thickness + c.core_thickness,
    ],
)
plt.colorbar(label="$log10(|n_{doped} - n_{undoped}|)$")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.ylim(1.72e-6, 2.5e-6)
plt.title("Voltage = 0V")

plt.imshow(
    np.log(np.abs(np.real(n_dist[-4].T - n_undoped.T))),
    origin="lower",
    extent=[
        -c.xmargin - c.ppp_offset - c.core_width / 2,
        c.xmargin + c.npp_offset + c.core_width / 2,
        0,
        c.clad_thickness + c.box_thickness + c.core_thickness,
    ],
)
plt.colorbar(label="$log10(|n_{doped} - n_{undoped}|)$")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.ylim(1.72e-6, 2.5e-6)
plt.title("Voltage = -4V")

plt.imshow(
    np.log(np.abs(np.imag(n_dist[0].T - n_undoped.T))),
    origin="lower",
    extent=[
        -c.xmargin - c.ppp_offset - c.core_width / 2,
        c.xmargin + c.npp_offset + c.core_width / 2,
        0,
        c.clad_thickness + c.box_thickness + c.core_thickness,
    ],
)
plt.colorbar(label=r"$log10(|\kappa_{doped} - \kappa_{undoped}|)$")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.ylim(1.72e-6, 2.5e-6)
plt.title("Voltage = 0V")

plt.imshow(
    np.log(np.abs(np.imag(n_dist[-4].T))),
    origin="lower",
    extent=[
        -c.xmargin - c.ppp_offset - c.core_width / 2,
        c.xmargin + c.npp_offset + c.core_width / 2,
        0,
        c.clad_thickness + c.box_thickness + c.core_thickness,
    ],
)
plt.colorbar(label=r"$log10(|\kappa_{doped} - \kappa_{undoped}|)$")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.ylim(1.72e-6, 2.5e-6)
plt.title("Voltage = -4V")
