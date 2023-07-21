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

# +
import numpy as np
import gdsfactory.simulation.gtidy3d as gt
import matplotlib.pyplot as plt
import gdsfactory as gf

gf.config.rich_output()
PDK = gf.generic_tech.get_generic_pdk()
PDK.activate()

nm = 1e-3
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

strip.plot_grid()

strip.plot_field(field_name="Ex", mode_index=0)  # TE

strip.plot_field(field_name="Ex", mode_index=0, value="dB")  # TE

strip.plot_field(field_name="Ey", mode_index=1)  # TM

strip.n_eff

rib = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=0.5,
    core_thickness=0.22,
    slab_thickness=0.15,
    core_material="si",
    clad_material="sio2",
)
rib.plot_index()
rib.n_eff

rib.plot_field(field_name="Ex", mode_index=0)  # TE

nitride = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=1.0,
    core_thickness=400 * nm,
    slab_thickness=0.0,
    core_material="sin",
    clad_material="sio2",
)
nitride.plot_index()
nitride.n_eff

nitride.plot_field(field_name="Ex", mode_index=0)  # TE

# ## Sweep width
#
# You can sweep the waveguide width and compute the modes.
#
# By increasing the waveguide width, the waveguide supports many more TE and TM modes. Where TE modes have a dominant Ex field and TM modes have larger Ey fields.
#
# Notice that waveguides wider than 0.450 um support more than one TE mode. Therefore the maximum width for single mode operation is 0.450 um.
#

# +
strip = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=1.0,
    slab_thickness=0.0,
    core_material="si",
    clad_material="sio2",
    core_thickness=220 * nm,
    num_modes=4,
)
w = np.linspace(400 * nm, 1000 * nm, 7)
n_eff = gt.modes.sweep_n_eff(strip, core_width=w)
fraction_te = gt.modes.sweep_fraction_te(strip, core_width=w)

for i in range(4):
    plt.plot(w, n_eff.sel(mode_index=i).real, c="k")
    plt.scatter(
        w, n_eff.sel(mode_index=i).real, c=fraction_te.sel(mode_index=i), vmin=0, vmax=1
    )
plt.axhline(y=1.44, color="k", ls="--")
plt.colorbar().set_label("TE fraction")
plt.xlabel("Width of waveguide (µm)")
plt.ylabel("Effective refractive index")
plt.title("Effective index sweep")
# -

# **Exercises**
#
# - What is the maximum width to support a single TE mode at 1310 nm?
# - For a Silicon Nitride (n=2) 400nm thick waveguide surrounded by SiO2 (n=1.44), what is the maximum width to support a single TE mode at 1550 nm?
# - For two 500x220nm Silicon waveguides surrounded by SiO2, what is the coupling length (100% coupling) for 200 nm gap?
#

# ## Group index
#
# You can also compute the group index for a waveguide.

# +
nm = 1e-3

strip = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=500 * nm,
    slab_thickness=0.0,
    core_material="si",
    clad_material="sio2",
    core_thickness=220 * nm,
    num_modes=4,
    group_index_step=10 * nm,
)
print(strip.n_group)
# -

# ## Bend modes
#
# You can compute bend modes specifying the bend radius.

strip_bend = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=500 * nm,
    core_thickness=220 * nm,
    slab_thickness=0.0,
    bend_radius=4,
    core_material="si",
    clad_material="sio2",
)
strip_bend.plot_field(field_name="Ex", mode_index=0)  # TE

# ## Bend loss
#
# You can also compute the losses coming from the mode mismatch from the bend into a straight waveguide.
# To compute the bend loss due to mode mismatch you can calculate the mode overlap of the straight mode and the bent mode.
# Because there are two mode mismatch interfaces the total loss due to mode mismatch will be squared (from bend to straight and from straight to bend).
#
# ![](https://i.imgur.com/M1Yysdr.png)
#
# [from paper](https://ieeexplore.ieee.org/ielaam/50/8720127/8684870-aam.pdf)

# +
radii = np.arange(4, 7)
bend = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=500 * nm,
    core_thickness=220 * nm,
    core_material="si",
    clad_material="sio2",
    num_modes=1,
    bend_radius=radii.min(),
)
mismatch = gt.modes.sweep_bend_mismatch(bend, radii)

plt.plot(radii, 10 * np.log10(mismatch))
plt.title("Strip waveguide bend")
plt.xlabel("Radius (μm)")
plt.ylabel("Mismatch (dB)")


# +
dB_cm = 2  # dB/cm
length = 2 * np.pi * radii * 1e-6
propagation_loss = dB_cm * length * 1e2
propagation_loss

plt.title("Bend90 loss for TE polarization")
plt.plot(radii, -10 * np.log10(mismatch), ".", label="mode loss")
plt.plot(radii, propagation_loss, ".", label="propagation loss")
plt.xlabel("bend radius (um)")
plt.ylabel("Loss (dB)")
plt.legend()
# -

rib = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=1000 * nm,
    core_thickness=220 * nm,
    slab_thickness=110 * nm,
    bend_radius=15,
    core_material="si",
    clad_material="sio2",
)
rib.plot_field(field_name="Ex", mode_index=0)  # TE

nitride_bend = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=1000 * nm,
    core_thickness=400 * nm,
    slab_thickness=0.0,
    bend_radius=30,
    core_material="sin",
    clad_material="sio2",
)
nitride_bend.plot_field(field_name="Ex", mode_index=0, value="abs")  # TE

radii = np.array([30, 35, 40])
bend = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=1000 * nm,
    core_thickness=400 * nm,
    core_material="sin",
    clad_material="sio2",
    num_modes=1,
    bend_radius=radii.min(),
)
mismatch = gt.modes.sweep_bend_mismatch(bend, radii)


# +
dB_cm = 2  # dB/cm
length = 2 * np.pi * radii * 1e-6
propagation_loss = dB_cm * length * 1e2
propagation_loss

plt.title("Bend90 loss for TE polarization")
plt.plot(radii, -10 * np.log10(mismatch), ".", label="mode loss")
plt.plot(radii, propagation_loss, ".", label="propagation loss")
plt.xlabel("bend radius (um)")
plt.ylabel("Loss (dB)")
plt.legend()
# -

# **Exercises**
#
# - For a 500nm wide 220nm thick Silicon waveguide surrounded by SiO2, what is the minimum bend radius to have less than 0.04dB loss for TE polarization at 1550nm?
# - For a 500nm wide 220nm thick Silicon waveguide surrounded by SiO2, what is the minimum bend radius to have 99% power transmission for TM polarization at 1550nm?

# ## Waveguide coupler
#
# You can also compute the modes of a waveguide coupler.
#
# ```
#        ore_width[0]  core_width[1]
#         <------->     <------->
#          _______       _______   _
#         |       |     |       | |
#         |       |     |       |
#         |       |_____|       | | core_thickness
#         |slab_thickness       |
#         |_____________________| |_
#                 <----->
#                   gap
#
#
# ```

c = gt.modes.WaveguideCoupler(
    wavelength=1.55,
    core_width=(500 * nm, 500 * nm),
    gap=200 * nm,
    core_thickness=220 * nm,
    slab_thickness=100 * nm,
    core_material="si",
    clad_material="sio2",
)
c.plot_index()

c.plot_field(field_name="Ex", mode_index=0)  # TE

c.plot_field(field_name="Ex", mode_index=1)  # TE

# +
coupler = gt.modes.WaveguideCoupler(
    wavelength=1.55,
    core_width=(450 * nm, 450 * nm),
    core_thickness=220 * nm,
    core_material="si",
    clad_material="sio2",
    num_modes=4,
    gap=0.1,
)

print("\nCoupler:", coupler)
print("Effective indices:", coupler.n_eff)
print("Mode areas:", coupler.mode_area)
print("Coupling length:", coupler.coupling_length())

gaps = np.linspace(0.05, 0.15, 11)
lengths = gt.modes.sweep_coupling_length(coupler, gaps)

_, ax = plt.subplots(1, 1)
ax.plot(gaps, lengths)
ax.set(xlabel="Gap (μm)", ylabel="Coupling length (μm)")
ax.legend(["TE", "TM"])
ax.grid()
