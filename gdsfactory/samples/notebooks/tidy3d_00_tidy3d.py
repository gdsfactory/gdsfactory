# # FDTD tidy3d
#
# [tidy3D](https://docs.flexcompute.com/projects/tidy3d/en/latest/) is a fast GPU based FDTD tool developed by flexcompute.
#
# To run, you need to [create an account](https://simulation.cloud/) and add credits. The number of credits that each simulation takes depends on the simulation computation time.
#
# We have commented the `write_sparameters` functions to save on credits when running these simulations.
#
# ![cloud_model](https://i.imgur.com/5VTCPLR.png)
#
# ## Materials
#
# Tidy3d provides you with a material database of dispersive materials.

# +
import matplotlib.pyplot as plt
import numpy as np
from gdsfactory.components.taper import taper_sc_nc
import gdsfactory.simulation as sim
import gdsfactory.simulation.gtidy3d as gt
import gdsfactory as gf
from gdsfactory.config import PATH
import tidy3d as td

gf.config.rich_output()
PDK = gf.generic_tech.get_generic_pdk()
PDK.activate()
# -

nm = 1e-3
wavelength = np.linspace(1500, 1600) * nm
f = td.C_0 / wavelength
eps_complex = td.material_library["cSi"]["Li1993_293K"].eps_model(f)
n, k = td.Medium.eps_complex_to_nk(eps_complex)
plt.plot(wavelength, n)
plt.title("cSi crystalline silicon")
plt.xlabel("wavelength")
plt.ylabel("n")

eps_complex = td.material_library["Si3N4"]["Luke2015PMLStable"].eps_model(f)
n, k = td.Medium.eps_complex_to_nk(eps_complex)
plt.plot(wavelength, n)
plt.title("SiN")
plt.xlabel("wavelength")
plt.ylabel("n")

eps_complex = td.material_library["SiO2"]["Horiba"].eps_model(f)
n, k = td.Medium.eps_complex_to_nk(eps_complex)
plt.plot(wavelength, n)
plt.title("SiO2")
plt.xlabel("wavelength")
plt.ylabel("n")

# ## get_simulation
#
# You can run `get_simulation` to convert a gdsfactory planar Component into a tidy3d simulation and make sure the simulation looks correct before running it
#
# `get_simulation` also has a `plot_modes` option so you can make sure you are monitoring the desired mode.

# ### 2D
#
# 2D planar simulations run faster than 3D. When running in 2D we don't consider the component thickness in the z dimension

c = gf.components.mmi1x2()
s = gt.get_simulation(c, is_3d=False)
fig = gt.plot_simulation(s)

# ### 3D
#
# By default all simulations run in 3D unless indicated otherwise with the `is_3d` argument.
# 3D simulations run quite fast thanks to the GPU solver on the server side hosted by tidy3d cloud.

help(gt.get_simulation)

c = gf.components.mmi1x2()
s = gt.get_simulation(c)
fig = gt.plot_simulation(s)

c = gf.components.coupler_ring()
s = gt.get_simulation(c)
fig = gt.plot_simulation(s)

c = gf.components.bend_circular(radius=2)
s = gt.get_simulation(c)
fig = gt.plot_simulation(s)

c = gf.components.straight()
s = gt.get_simulation(c)
fig = gt.plot_simulation(s)

# ## Sidewall angle
#
# You can define the sidewall angle in degrees with respect to normal. Lets exaggerate the sidewall angle so we can clearly see it.

c = gf.components.straight()
s = gt.get_simulation(c, sidewall_angle_deg=45, plot_modes=True)
fig = gt.plot_simulation(s)

# ## Erosion / dilation

c = gf.components.straight()
s = gt.get_simulation(c, is_3d=False, dilation=0)
fig = gt.plot_simulation(s)

c = gf.components.straight()
s = gt.get_simulation(c, is_3d=False, dilation=0.5)
fig = gt.plot_simulation(s)

0.5 * 1.5

# A `dilation = 0.5` makes a 0.5um waveguide 0.75um

0.5 * 0.8

# A `dilation = -0.2` makes a 0.5um eroded down to 0.1um

0.2 * 0.5

c = gf.components.straight()
s = gt.get_simulation(c, is_3d=False, dilation=-0.2)
fig = gt.plot_simulation(s)

# ## Plot source and monitor modes

c = gf.components.straight(length=3)
s = gt.get_simulation(c, plot_modes=True, port_margin=1, ymargin=1)
fig = gt.plot_simulation_xz(s)

c = gf.components.straight_rib(length=3)
s = gt.get_simulation(c, plot_modes=True)
fig = gt.plot_simulation_xz(s)

c = taper_sc_nc(length=10)
s = gt.get_simulation(c, plot_modes=True)
fig = gt.plot_simulation_xz(s)

# Lets make sure the mode also looks correct on the Nitride side

c = taper_sc_nc(length=10)
s = gt.get_simulation(c, port_source_name="o2", plot_modes=True)
fig = gt.plot_simulation_xz(s)

# +
components = [
    "bend_euler",
    "bend_s",
    "coupler",
    "coupler_ring",
    "crossing",
    "mmi1x2",
    "mmi2x2",
    "taper",
    "straight",
]

for component_name in components:
    print(component_name)
    plt.figure()
    c = gf.components.cells[component_name]()
    s = gt.get_simulation(c)
    fig = gt.plot_simulation(s)
# -

# ## write_sparameters
#
# You can write Sparameters from a simulation as well as a group of simulations in parallel.

c = gf.components.bend_circular(radius=2)
s = gt.get_simulation(c)
fig = gt.plot_simulation(s)

# For a 2 port reciprocal passive component you can always assume `s21 = s12`
#
# Another approximation you can make for planar devices is that `s11 = s22`, which saves 1 extra simulation.
# This approximation only works well for straight and bends.
# We call this `1x1` port symmetry

# sp = gt.write_sparameters_1x1(c)
sp = np.load(
    PATH.sparameters_repo / "bend_circular_radius2_9d7742b34c224827aeae808dc986308e.npz"
)
sim.plot.plot_sparameters(sp)

sim.plot.plot_sparameters(sp, keys=("o2@0,o1@0",))

c = gf.components.mmi1x2()
s = gt.get_simulation(c, plot_modes=True, port_margin=0.2, port_source_name="o2")
fig = gt.plot_simulation(s, y=0)  # see input

fig = gt.plot_simulation(s, y=0.63)  # see output

# +
# sp = gt.write_sparameters(c)
# -

sp = np.load(PATH.sparameters_repo / "mmi1x2_507de731d50770de9096ac9f23321daa.npz")

sim.plot.plot_sparameters(sp)

sim.plot.plot_sparameters(sp, keys=("o1@0,o2@0", "o1@0,o3@0"))

sim.plot.plot_loss1x2(sp)

sim.plot.plot_imbalance1x2(sp)

c = gf.components.mmi2x2_with_sbend(with_sbend=False)
c.plot()

sp = gt.write_sparameters(c, run=False)

# sp = gt.write_sparameters(c, filepath=PATH.sparameters_repo / 'mmi2x2_without_sbend.npz')
sp = np.load(PATH.sparameters_repo / "mmi2x2_without_sbend.npz")
sim.plot.plot_loss2x2(sp)

sim.plot.plot_imbalance2x2(sp)

# ## write_sparameters_batch
#
# You can also send a batch of component simulations in parallel to the tidy3d server.
#
#
# ```python
# jobs = [dict(component=gf.c.straight(length=1.11 + i)) for i in [1, 2]]
# sps = gt.write_sparameters_batch_1x1(jobs)
#
# sp0 = sps[0]
# sp = sp0.result()
# sim.plot.plot_sparameters(sp)
# ```

# ## get_simulation_grating_coupler
#
# You can also expand the planar component simulations to simulate an out-of-plane grating coupler.
#
# The following simulations run in 2D but can also run in 3D.

help(gt.get_simulation_grating_coupler)

c = (
    gf.components.grating_coupler_elliptical_lumerical()
)  # inverse design grating apodized
fiber_angle_deg = 5
s = gt.get_simulation_grating_coupler(
    c, is_3d=False, fiber_angle_deg=fiber_angle_deg, fiber_xoffset=0
)
f = gt.plot_simulation(s)

f = c.plot()

# Lets compare the xtolerance of a constant pitch vs an apodized grating.
#
# We run simulations in 2D for faster.
#
# Lets simulate 2 different grating couplers:
#
# - apodized inverse design example from lumerical website (5 degrees fiber angle)
# - constant pitch grating from gdsfactory generic PDK (20 degrees fiber angle)

sim = gt.get_simulation_grating_coupler(
    c, is_3d=False, fiber_angle_deg=fiber_angle_deg, fiber_xoffset=-5
)
f = gt.plot_simulation(sim)

sim = gt.get_simulation_grating_coupler(
    c, is_3d=False, fiber_angle_deg=fiber_angle_deg, fiber_xoffset=+5
)
f = gt.plot_simulation(sim)

offsets = np.arange(-5, 6, 5)
offsets = [-10, -5, 0]
offsets = [0]

dfs = [
    gt.write_sparameters_grating_coupler(
        component=c,
        is_3d=False,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xoffset=fiber_xoffset,
        filepath=PATH.sparameters_repo / f"gc_offset{fiber_xoffset}",
    )
    for fiber_xoffset in offsets
]


def log(x):
    return 20 * np.log10(x)


# +
for offset in offsets:
    sp = gt.write_sparameters_grating_coupler(
        c,
        is_3d=False,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xoffset=offset,
        filepath=PATH.sparameters_repo / f"gc_offset{offset}",
    )
    plt.plot(
        sp["wavelengths"], 20 * np.log10(np.abs(sp["o2@0,o1@0"])), label=str(offset)
    )

plt.xlabel("wavelength (um")
plt.ylabel("Transmission (dB)")
plt.title("transmission vs fiber xoffset (um)")
plt.legend()
# -

sp.keys()

fiber_angles = [3, 5, 7]
dfs = [
    gt.write_sparameters_grating_coupler(
        component=c,
        is_3d=False,
        fiber_angle_deg=fiber_angle_deg,
        filepath=PATH.sparameters_repo / f"gc_angle{fiber_angle_deg}",
    )
    for fiber_angle_deg in fiber_angles
]

# +
for fiber_angle_deg in fiber_angles:
    sp = gt.write_sparameters_grating_coupler(
        c,
        is_3d=False,
        fiber_angle_deg=fiber_angle_deg,
        filepath=PATH.sparameters_repo / f"gc_angle{fiber_angle_deg}",
    )
    plt.plot(
        sp["wavelengths"],
        20 * np.log10(np.abs(sp["o2@0,o1@0"])),
        label=str(fiber_angle_deg),
    )

plt.xlabel("wavelength (um")
plt.ylabel("Transmission (dB)")
plt.title("transmission vs fiber angle (degrees)")
plt.legend()
# -

c = gf.components.grating_coupler_elliptical_arbitrary(
    widths=[0.343] * 25, gaps=[0.345] * 25
)
f = c.plot()

fiber_angle_deg = 20
sim = gt.get_simulation_grating_coupler(
    c, is_3d=False, fiber_angle_deg=fiber_angle_deg, fiber_xoffset=0
)
f = gt.plot_simulation(sim, figsize=(22, 8))

offsets = [0]
offsets

dfs = [
    gt.write_sparameters_grating_coupler(
        component=c,
        is_3d=False,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xoffset=fiber_xoffset,
        filepath=PATH.sparameters_repo / f"gc_offset{offset}",
    )
    for fiber_xoffset in offsets
]

# +
port_name = c.get_ports_list()[1].name

for offset in offsets:
    sp = gt.write_sparameters_grating_coupler(
        c,
        is_3d=False,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xoffset=offset,
        filepath=PATH.sparameters_repo / f"gc_offset{offset}",
    )
    plt.plot(
        sp["wavelengths"],
        20 * np.log10(np.abs(sp["o2@0,o1@0"])),
        label=str(offset),
    )

plt.xlabel("wavelength (um")
plt.ylabel("Transmission (dB)")
plt.title("transmission vs xoffset")
plt.legend()
# -

# ## Run jobs in parallel
#
# You can run multiple simulations in parallel on separate threads.
#
# Only when you `sp.result()` you will wait for the simulations to finish.

c = gf.components.grating_coupler_elliptical_lumerical()
fiber_angles = [3, 5, 7]
jobs = [
    dict(
        component=c,
        is_3d=False,
        fiber_angle_deg=fiber_angle_deg,
        filepath=PATH.sparameters_repo / f"gc_angle{fiber_angle_deg}",
    )
    for fiber_angle_deg in fiber_angles
]
sps = gt.write_sparameters_grating_coupler_batch(jobs)

# +
for sp, fiber_angle_deg in zip(sps, fiber_angles):
    sp = sp.result()
    plt.plot(
        sp["wavelengths"],
        20 * np.log10(np.abs(sp["o2@0,o1@0"])),
        label=str(fiber_angle_deg),
    )

plt.xlabel("wavelength (um")
plt.ylabel("Transmission (dB)")
plt.title("transmission vs fiber angle (degrees)")
plt.legend()
# -

bend_radius = [1, 2]
jobs = [
    dict(
        component=gf.components.bend_circular(radius=radius),
        filepath=PATH.sparameters_repo / f"bend_r{radius}",
    )
    for radius in bend_radius
]
sps = gt.write_sparameters_batch(jobs)

# +
for sp, radius in zip(sps, bend_radius):
    sp = sp.result()
    plt.plot(
        sp["wavelengths"],
        20 * np.log10(np.abs(sp["o2@0,o1@0"])),
        label=str(radius),
    )

plt.xlabel("wavelength (um")
plt.ylabel("Transmission (dB)")
plt.title("transmission vs bend radius (um)")
plt.legend()
