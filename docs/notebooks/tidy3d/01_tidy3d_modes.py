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
# # Tidy3D mode solver
#
# Tidy3d comes with an open source FDFD [mode solver](https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/ModeSolver.html)
#
# ## Waveguides
#
# Guided Electromagnetic modes are the ones that have an effective index larger than the cladding of the waveguide
#
# Here is a waveguide of Silicon (n=3.4) surrounded by SiO2 (n=1.44) cladding
#
# For a 220 nm height x 450 nm width the effective index is 2.466

# %% tags=[]
import numpy as np
import gdsfactory.simulation.gtidy3d as gt
import matplotlib.pyplot as plt
import gdsfactory as gf

gf.config.rich_output()
PDK = gf.generic_tech.get_generic_pdk()
PDK.activate()

nm = 1e-3

# %%
strip = gt.modes.Waveguide(
    wavelength=1.55,
    wg_width=0.5,
    wg_thickness=0.22,
    slab_thickness=0.0,
    ncore="si",
    nclad="sio2",
)
strip.plot_index()

# %%
strip.plot_Ex(0)  # TE
strip.plot_Ey(1)  # TM

# %%
strip.neffs[0].real

# %%
rib = gt.modes.Waveguide(
    wavelength=1.55,
    wg_width=0.5,
    wg_thickness=0.22,
    slab_thickness=0.15,
    ncore="si",
    nclad="sio2",
)
rib.plot_index()

# %%
rib.plot_Ex(mode_index=0)
rib.plot_Ey(mode_index=0)

# %%
nitride = gt.modes.Waveguide(
    wavelength=1.55,
    wg_width=1.0,
    wg_thickness=0.4,
    slab_thickness=0.0,
    ncore="si",
    nclad="sio2",
)
nitride.plot_index()
nitride.plot_Ex(0)
nitride.plot_Ey(0)

# %% [markdown]
# ## Sweep width
#
# You can sweep the waveguide width and compute the modes.
#
# By increasing the waveguide width, the waveguide supports many more TE and TM modes. Where TE modes have a dominant Ex field and TM modes have larger Ey fields.
#
# Notice that waveguides wider than 0.450 um support more than one TE mode. Therefore the maximum width for single mode operation is 0.450 um.
#

# %%
df = gt.modes.sweep_width(
    width1=200 * nm,
    width2=1000 * nm,
    steps=11,
    wavelength=1.55,
    wg_thickness=220 * nm,
    slab_thickness=0 * nm,
    ncore="si",
    nclad="sio2",
)
gt.modes.plot_sweep_width(
    width1=200 * nm,
    width2=1000 * nm,
    steps=11,
    wavelength=1.55,
    wg_thickness=220 * nm,
    slab_thickness=0 * nm,
    ncore="si",
    nclad="sio2",
)
plt.axhline(y=1.44, color="k", linestyle="--")

# %% [markdown]
# **Exercises**
#
# - What is the maximum width to support a single TE mode at 1310 nm?
# - For a Silicon Nitride (n=2) 400nm thick waveguide surrounded by SiO2 (n=1.44), what is the maximum width to support a single TE mode at 1550 nm?
# - For two 500x220nm Silicon waveguides surrounded by SiO2, what is the coupling length (100% coupling) for 200 nm gap?
#

# %% [markdown]
# ## Group index
#
# You can also compute the group index for a waveguide.

# %%
nm = 1e-3

ng = gt.modes.group_index(
    wg_width=500 * nm,
    wavelength=1.55,
    wg_thickness=220 * nm,
    slab_thickness=0 * nm,
    ncore="si",
    nclad="sio2",
)
print(ng)

# %%
nm = 1e-3
wg_widths = np.array([490, 500, 510]) * nm
wg_settings = dict(
    wg_thickness=220 * nm,
    slab_thickness=0 * nm,
    ncore="si",
    nclad="sio2",
)

ng = [
    gt.modes.group_index(wavelength=1550 * nm, wg_width=wg_width, **wg_settings)
    for wg_width in wg_widths
]
plt.plot(wg_widths * 1e3, ng)
plt.xlabel("waveguide width (nm)")
plt.ylabel("ng")

# %% [markdown]
# ## Bend modes
#
# You can compute bend modes specifying the bend radius.

# %%
strip_bend = gt.modes.Waveguide(
    wavelength=1.55,
    wg_width=0.5,
    wg_thickness=0.22,
    slab_thickness=0.0,
    bend_radius=3,
    ncore="si",
    nclad="sio2",
)

# %%
# plot the fundamental TE mode
strip_bend.plot_Ex(0)
strip_bend.plot_Ey(0)

# %%
# plot the fundamental TM mode
strip_bend.plot_Ex(1)
strip_bend.plot_Ey(1)

# %% [markdown]
# ## Bend loss
#
# You can also compute the losses coming from the mode mismatch from the bend into a straight waveguide.
# To compute the bend loss due to mode mismatch you can calculate the mode overlap of the straight mode and the bent mode.
# Because there are two mode mismatch interfaces the total loss due to mode mismatch will be squared (from bend to straight and from straight to bend).
#
# ![](https://i.imgur.com/M1Yysdr.png)
#
# [from paper](https://ieeexplore.ieee.org/ielaam/50/8720127/8684870-aam.pdf)

# %%
r, integral = gt.modes.sweep_bend_loss(
    wavelength=1.55,
    wg_width=0.5,
    wg_thickness=0.22,
    slab_thickness=0.0,
    bend_radius_min=2.0,
    bend_radius_max=5,
    steps=4,
    mode_index=0,
    ncore="si",
    nclad="sio2",
)

plt.title("Bend90 loss for TE polarization")
plt.plot(r, integral, ".")
plt.xlabel("bend radius (um)")
plt.ylabel("Transmission")
plt.show()

# %%
dB_cm = 2  # dB/cm
length = 2 * np.pi * r * 1e-6
propagation_loss = dB_cm * length * 1e2
propagation_loss

# %%
plt.title("Bend90 loss for TE polarization")
plt.plot(r, -10 * np.log10(integral), ".", label="mode loss")
plt.plot(r, propagation_loss, ".", label="propagation loss")
plt.xlabel("bend radius (um)")
plt.ylabel("Loss (dB)")
plt.legend()

# %%
r, integral = gt.modes.sweep_bend_loss(
    wavelength=1.55,
    wg_width=0.5,
    wg_thickness=0.22,
    slab_thickness=0.0,
    bend_radius_min=3.0,
    bend_radius_max=20,
    steps=4,
    mode_index=1,
    ncore="si",
    nclad="sio2",
)

plt.title("Bend90 loss for TM polarization")
plt.ylim(ymin=min(integral), ymax=1)
plt.plot(r, integral, ".")
plt.xlabel("bend radius (um)")
plt.ylabel("Transmission")
plt.show()

# %%
dB_cm = 1  # dB/cm
length = 2 * np.pi * r * 1e-6
propagation_loss = dB_cm * length * 1e2
propagation_loss

# %%
plt.plot(r, -10 * np.log10(integral), ".", label="mode loss")
plt.plot(r, propagation_loss, ".", label="propagation loss")
plt.title("Bend90 loss for TM polarization")
plt.xlabel("bend radius (um)")
plt.ylabel("Loss (dB)")
plt.legend()

# %% [markdown]
# **Exercises**
#
# - For a 500nm wide 220nm thick Silicon waveguide surrounded by SiO2, what is the minimum bend radius to have less than 0.04dB loss for TE polarization at 1550nm?
# - For a 500nm wide 220nm thick Silicon waveguide surrounded by SiO2, what is the minimum bend radius to have 99% power transmission for TM polarization at 1550nm?

# %% [markdown]
# ## Waveguide coupler
#
# You can also compute the modes of a waveguide coupler.
#
# ```
#         wg_width1     wg_width2
#         <------->     <------->
#          _______       _______   _
#         |       |     |       | |
#         |       |     |       |
#         |       |_____|       | | wg_thickness
#         |slab_thickness       |
#         |_____________________| |_
#                 <----->
#                   gap
#
#
# ```

# %%
c = gt.modes.WaveguideCoupler(
    wavelength=1.55,
    wg_width1=500 * nm,
    wg_width2=500 * nm,
    gap=200 * nm,
    wg_thickness=220 * nm,
    slab_thickness=100 * nm,
    ncore="si",
    nclad="sio2",
)
c.plot_index()

# %%
c.plot_Ex(0, plot_power=False)  # even
c.plot_Ex(1, plot_power=False)  # odd

# %%
c = gt.modes.WaveguideCoupler(
    wavelength=1.55,
    wg_width1=500 * nm,
    wg_width2=500 * nm,
    gap=200 * nm,
    wg_thickness=220 * nm,
    slab_thickness=0 * nm,
    ncore="si",
    nclad="sio2",
)
c.plot_index()

# %%
c.plot_Ex(0, plot_power=False)  # even
c.plot_Ex(1, plot_power=False)  # odd

# %%
nm = 1e-3
si = gt.modes.si
sio2 = gt.modes.sio2
c = gt.modes.WaveguideCoupler(
    wavelength=1.55,
    wg_width1=500 * nm,
    wg_width2=500 * nm,
    gap=200 * nm,
    wg_thickness=220 * nm,
    slab_thickness=0 * nm,
    ncore="si",
    nclad="sio2",
)
c.plot_index()

# %%
gaps = [150, 200, 250, 300]
coupling_length = [
    gt.modes.WaveguideCoupler(
        wavelength=1.55,
        wg_width1=500 * nm,
        wg_width2=500 * nm,
        gap=gap * nm,
        wg_thickness=220 * nm,
        slab_thickness=0 * nm,
        ncore="si",
        nclad="sio2",
    ).find_coupling(power_ratio=1)
    for gap in gaps
]

# %%
plt.plot(gaps, coupling_length, ".")
plt.xlabel("gap (nm)")
plt.ylabel("100% coupling length (um)")

# %% tags=[]
df = gt.modes.find_coupling_vs_gap(
    wg_width1=500 * nm,
    wg_width2=500 * nm,
    wg_thickness=220 * nm,
    slab_thickness=0 * nm,
    ncore="si",
    nclad="sio2",
)
df
