# # MPB mode-solver
#
# [MPB](https://mpb.readthedocs.io/en/latest/Python_Tutorial/#our-first-band-structure) is a free open source software to compute:
#
# - electro-magnetic modes
# - band structures
#
# supported by a waveguide with periodic boundaries.
#
#
# ## Find modes waveguide
#
# Lets find the modes supported by a waveguide for a particular waveguide geometry and wavelength.
#
# A waveguide is like a pipe to guide the light and is made of a higher refractive index core `core_material` surrounded by a lower refractive index cladding `clad_material`
#
#
# ```bash
#           __________________________
#           |
#           |
#           |         width
#           |     <---------->
#           |      ___________   _ _ _
#           |     |           |       |
#         sz|_____|           |_______|
#           |                         | core_thickness
#           |slab_thickness           |
#           |_________________________|
#           |
#           |
#           |__________________________
#           <------------------------>
#                         sy
# ```
#
# Silicon is yellow and opaque at visible wavelengths (380 to 700nm). This is the reason why CMOS cameras can be made of Silicon.
#
# At Infra-red wavelengths used for communications (1300 or 1550nm) Silicon is transparent and has a high refractive index `3.47`. So making a Silicon waveguide is quite easy, where the Silicon is the guiding material, and Silicon oxide `n=1.45` makes a great low index material for the cladding of the waveguide.
#
#
# This [video](https://www.youtube.com/watch?v=Hy7yn2xohlE) explains how Silicon Photonic waveguides guide light in Photonic integrated circuits.
#
#
#
# ### Strip waveguides
#
# Strip waveguides are fully etch and don't have a slab. `slab_thickness = 0`
#
#
# ```bash
#           __________________________
#           |
#           |
#           |         width
#           |     <---------->
#           |      ___________   _ _ _
#           |     |           |       |
#         sz|     |           |       |
#           |     |  core_material    | core_thickness
#           |     |           |       |
#           |     |___________|  _ _ _|
#           |
#           |        clad_material
#           |__________________________
#           <------------------------>
#                         sy
# ```

# +
import pandas as pd
import pathlib
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import gdsfactory.simulation.modes as gm
import gdsfactory as gf

from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()
# -

modes = gm.find_modes_waveguide(
    parity=mp.NO_PARITY,
    core_width=0.4,
    core_material=3.47,
    clad_material=1.44,
    core_thickness=0.22,
    resolution=40,
    sy=3,
    sz=3,
    nmodes=4,
)
m1 = modes[1]
m2 = modes[2]
m3 = modes[3]

# As you can see the refractive index goes from 1.44 `SiO2` Silicon dioxide to 3.47 `Silicon`.

m1.plot_eps()

m1.neff

m1.plot_ey()

m2.plot_e_all()

m1.plot_e()

# As you can see the first order mode has most power in y-direction `Ey`. This type of mode is called TE (transverse-electric)
#
# On the other hand the second order mode has most of the light in the `Ex`. This mode is called TM (transverse-magnetic)

m2.plot_e_all()

m3.plot_e()  # not guided

m1.neff

m2.neff

# the third mode does not propagate and its neff is below the cladding index
m3.neff

# ### Sidewall angle
#
# You can also specify the sidewall angle.

modes = gm.find_modes_waveguide(
    parity=mp.NO_PARITY,
    core_width=0.4,
    core_material=3.47,
    clad_material=1.44,
    core_thickness=0.22,
    resolution=40,
    sidewall_angle=10,
)
m1 = modes[1]
m2 = modes[2]
m3 = modes[3]
m1.plot_eps()

modes = gm.find_modes_waveguide(
    parity=mp.NO_PARITY,
    core_width=0.4,
    core_material=3.47,
    clad_material=1.44,
    core_thickness=0.22,
    resolution=60,
    sidewall_angle=10,
    slab_thickness=90e-3,
)
m1 = modes[1]
m2 = modes[2]
m3 = modes[3]
m1.plot_eps()

# ### Rib waveguides
#
# Rib waveguides have a slab (not fully etched)

modes = gm.find_modes_waveguide(
    mode_number=1, nmodes=2, slab_thickness=90e-3, resolution=40
)
m1 = modes[1]
m2 = modes[2]

m1.plot_eps()

m1.plot_e_all()

# ## Symmetries
#
# You can exploit symmetries to reduce computation time as well as finding only (TE or TM) modes
#
# MPB assumes propagation in the X direction
#
# - TE: mp.ODD_Y + mp.EVEN_Z
# - TM: mp.EVEN+Y + mp.ODD_Z, all energy in z component
#
# ### TM: mp.ODD_Y + mp.EVEN_Z
#
# You can define an even Y parity to find only the TM modes

modes = gm.find_modes_waveguide(
    mode_number=1,
    parity=mp.EVEN_Y + mp.ODD_Z,
    nmodes=2,
    core_width=1.0,
    core_material=3.47,
    clad_material=1.44,
    core_thickness=0.22,
    resolution=32,
    sy=6,
    sz=6,
)
m1 = modes[1]
m2 = modes[2]
m1.plot_e()

# ### ODD_Y (TE)

modes = gm.find_modes_waveguide(
    mode_number=1,
    parity=mp.ODD_Y,
    nmodes=2,
    core_width=0.20,
    core_material=3.47,
    clad_material=1.44,
    core_thickness=0.22,
    resolution=20,
    sy=5,
    sz=5,
)
m1 = modes[1]
m2 = modes[2]

m1.plot_e()

# ## Sweep waveguide width
#
# ### Strip

df = gm.find_neff_vs_width(filepath="data/mpb_neff_vs_width.csv")
df

gm.plot_neff_vs_width(df)

# ### Rib

modes = gm.find_modes_waveguide(
    core_width=0.4,
    core_material=3.47,
    clad_material=1.44,
    core_thickness=220e-3,
    resolution=20,
    sz=6,
    sy=6,
    nmodes=4,
    slab_thickness=90e-3,
)
m1 = modes[1]
m2 = modes[2]
m3 = modes[3]

m1.plot_eps()
m1.neff

m1.plot_e()
m1.neff

m2.plot_e()
m2.neff

df = gm.find_neff_vs_width(
    slab_thickness=90e-3, filepath="data/mpb_neff_vs_width_rib.csv"
)
gm.plot_neff_vs_width(df)

# ### Nitride

modes = gm.find_modes_waveguide(
    core_width=1.0,
    core_material=2.0,
    clad_material=1.44,
    core_thickness=400e-3,
    sz=6,
    sy=10,
    nmodes=4,
    resolution=10,
)
m1 = modes[1]
m2 = modes[2]
m3 = modes[3]

m1.plot_eps()

m1.plot_ey()

m1.plot_e_all()

m2.plot_ex()

m3.plot_ey()

df = gm.find_neff_vs_width(
    width1=0.5,
    width2=1.2,
    core_thickness=0.4,
    core_material=2.0,
    sy=10.0,
    resolution=15,
    filepath="data/mpb_neff_vs_width_nitride.csv",
)
gm.plot_neff_vs_width(df)

# ## Dispersion
#
# To get the effective index we only need to compute the mode propagation constant at a single frequency.
#
# However, to compute the dispersion (group delay) we need to compute the effective index for at least 3 wavelengths.
#
# The effective index `neff` relates to the speed of the phase evolution of the light, while the group index `ng` relates to the group velocity of the light.
#
# To compute the resonances in MZI interferometers or ring resonators you need to use `ng`

help(gm.find_mode_dispersion)

m = gm.find_mode_dispersion()

m.ng

# ## Convergence tests
#
# Before launching a set of simulations you need to make sure you have the correct simulation settings:
#
# - resolution: resolution
# - sx: Size of the simulation region in the x-direction (default=4.0)
# - sy: Size of the simulation region in the y-direction (default=4.0)
#

# +
resolutions = np.linspace(10, 50, 5)
neffs = []

for resolution in resolutions:
    modes = gm.find_modes_waveguide(
        core_width=0.5,
        core_material=3.5,
        clad_material=1.44,
        core_thickness=0.22,
        resolution=resolution,
    )
    mode = modes[1]
    neffs.append(mode.neff)
# -

plt.plot(resolutions, neffs, "o-")
plt.ylabel("neff")
plt.xlabel("resolution (pixels/um)")

# +
szs = np.linspace(4, 6, 6)
neffs = []

for sz in szs:
    modes = gm.find_modes_waveguide(
        core_width=0.5,
        core_material=3.5,
        clad_material=1.44,
        core_thickness=0.22,
        resolution=20,
        sz=sz,
    )
    mode = modes[1]
    neffs.append(mode.neff)
# -

plt.plot(szs, neffs, "o-")
plt.ylabel("neff")
plt.xlabel("simulation size in z(um)")

# +
sys = np.linspace(2, 6, 6)
neffs = []

for sy in sys:
    modes = gm.find_modes_waveguide(
        core_width=0.5,
        core_material=3.5,
        clad_material=1.44,
        core_thickness=0.22,
        resolution=20,
        sy=sy,
    )
    mode = modes[1]
    neffs.append(mode.neff)
# -

plt.plot(sys, neffs, "o-")
plt.ylabel("neff")
plt.xlabel("simulation size in y (um)")

# ## Find modes coupler
#
# When two waveguides are close to each other, they support modes that travel with different index (speed). One of the modes is an even mode, while the other one is an odd mode.
#
# Light will couple from one waveguide to another because the even and odd modes travel at different speeds and they interfere with each other. Creating a periodically back and forth coupling between both waveguides.
#
# Depending on the length of the coupling region and the gap there will be a different percentage of the light coupled from one to another
#
#
# ```bash
#
#           _____________________________________________________
#           |
#           |
#           |         widths[0]                 widths[1]
#           |     <---------->     gaps[0]    <---------->
#           |      ___________ <-------------> ___________      _
#           |     |           |               |           |     |
#         sz|_____|           |_______________|           |_____|
#           |    core_material                                  | core_thickness
#           |slab_thickness        nslab                        |
#           |___________________________________________________|
#           |
#           |<--->                                         <--->
#           |ymargin               clad_material                   ymargin
#           |____________________________________________________
#           <--------------------------------------------------->
#                                    sy
#
#
# ```

modes = gm.find_modes_coupler(
    core_widths=(0.5, 0.5),
    gaps=(0.2,),
    core_material=3.47,
    clad_material=1.44,
    core_thickness=0.22,
    resolution=20,
    sz=6,
    nmodes=4,
)
m1 = modes[1]
m2 = modes[2]
m3 = modes[3]

m1.plot_eps()

m1.plot_ey()  # even mode

m2.plot_ey()  # odd mode

# ### Find coupling vs gap

# +
# gm.find_coupling_vs_gap?

# +
df = gm.coupler.find_coupling_vs_gap(
    gap1=0.2,
    gap2=0.4,
    steps=12,
    nmodes=4,
    wavelength=1.55,
    filepath="data/mpb_find_coupling_vs_gap_strip.csv",
)

plt.title("strip 500x200 coupling")
gm.plot_coupling_vs_gap(df)

# +
df = gm.coupler.find_coupling_vs_gap_nitride(
    filepath="data/mpb_find_coupling_vs_gap_nitride.csv"
)

plt.title("nitride 1000x400 nitride")
gm.plot_coupling_vs_gap(df)

# +
ne = []
no = []
gaps = [0.2, 0.25, 0.3]

for gap in gaps:
    modes = gm.find_modes_coupler(
        core_widths=(0.5, 0.5),
        gaps=(gap,),
        core_material=3.47,
        clad_material=1.44,
        core_thickness=0.22,
        resolution=20,
        sz=6,
        nmodes=4,
    )
    ne.append(modes[1].neff)
    no.append(modes[2].neff)


# -


def coupling_length(
    neff1: float,
    neff2: float,
    power_ratio: float = 1.0,
    wavelength: float = 1.55,
) -> float:
    """Returns the coupling length (um) of the directional coupler
    to achieve power_ratio.

    Args:
        neff1: even supermode of the directional coupler.
        neff2: odd supermode of the directional coupler.
        power_ratio: p2/p1, where 1 means 100% power transfer.
        wavelength: in um.

    """
    dneff = (neff1 - neff2).real
    return wavelength / (np.pi * dneff) * np.arcsin(np.sqrt(power_ratio))


lc = [
    coupling_length(neff1=neff1, neff2=neff2) for gap, neff1, neff2 in zip(gaps, ne, no)
]

plt.plot(gaps, lc, ".-")
plt.ylabel("100% coupling length (um)")
plt.xlabel("gap (um)")

# ## Heater efficiency
#
# You can simulate the index change effect from a heater in MPB
#
# Lets assume the temperature increases by 10C (the actual increase does not matter)
#
# **Question**
#
# What is the optimal waveguide width for maximum index change?

dn_dt_si = 1.87e-4
dn_dt_sio2 = 8.5e-6

core_width = np.arange(0.4, 1.3, 0.2)
core_width

filepath = pathlib.Path("data/mpb_neff_vs_temperature.csv")

# +
if filepath.exists:
    df = pd.read_csv(filepath)
    dt = 10

else:
    dneffs = []
    for core_width in tqdm(core_width):
        dt = 0
        modes_t0 = gm.find_modes_waveguide(
            core_width=core_width,
            core_material=3.47 + dn_dt_si * dt,
            clad_material=1.44 + dn_dt_sio2 * dt,
            core_thickness=0.22,
            resolution=20,
            sy=6,
            sz=6,
            nmodes=4,
        )
        m1 = modes_t0[1]
        neff_t0 = m1.neff

        dt = 10
        modes_t1 = gm.find_modes_waveguide(
            core_width=core_width,
            core_material=3.47 + dn_dt_si * dt,
            clad_material=1.44 + dn_dt_sio2 * dt,
            core_thickness=0.22,
            resolution=20,
            sy=6,
            sz=6,
            nmodes=4,
        )
        m1 = modes_t1[1]
        neff_t1 = m1.neff

        dneff = neff_t1 - neff_t0
        dneffs.append(dneff)

    df = pd.DataFrame(dict(core_width=core_width, dneff=dneffs))
    df.to_csv(filepath)
# -

core_width = df.core_width
dneffs = df.dneff

plt.plot(core_width, np.array(dneffs) / dt, ".-")
plt.xlabel("waveguide width (um)")
plt.ylabel("dneff / dT")

dndt = np.array(dneffs) / dt
plt.plot(core_width, dndt / max(dndt) * 100, ".-")
plt.title("waveguide dn/dT")
plt.xlabel("waveguide width (um)")
plt.ylabel("dn/dT (%)")
