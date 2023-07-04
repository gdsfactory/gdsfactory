# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cascaded MZI Filter
#
# This example shows how to assemble components together to form a complex component that can be simulated by integrating `gdsfactory`, `tidy3d`, and `sax`.  The design is based on the first stage of the Coarse Wavelength Division Multiplexer presented in S. Dwivedi, P. De Heyn, P. Absil, J. Van Campenhout and W. Bogaerts, “Coarse wavelength division multiplexer on silicon-on-insulator for 100 GbE,” _2015 IEEE 12th International Conference on Group IV Photonics (GFP)_, Vancouver, BC, Canada, 2015, pp. 9-10, doi: [10.1109/Group4.2015.7305928](https://doi.org/10.1109/Group4.2015.7305928).
#
# Each filter stage is formed by 4 cascaded Mach-Zenhder Interferometers (MZIs) with predefined delays for the central wavelength.  Symmetrical Direction Couplers (DCs) are used to mix the signals at the ends of the MZI arms.  In order to facilitate fabrication, all DC gaps are kept equal, so the power transfer ratios are defined by the coupling length of the DCs.
#
# We will design each DC through 3D FDTD simulations to guarantee the desired power ratios, which have been calculated to provide maximally flat response.  The S parameters computed through FDTD are latter used in the full circuit simulation along with models for staight and curved waveguide sections, leading to an accurate model that exhibits features similar to those found in experimental data.

# %%
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import tidy3d as td
import sax

import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt
from gdsfactory.config import PATH


# %% [markdown]
# We start by loading the desired PDK and setting the main geometry and filter parameters, such as DC gap and central wavelength.

# %%
fsr = 0.01
gap = 0.15
width = 0.45
wavelengths = np.linspace(1.5, 1.6, 101)
lda_c = wavelengths[wavelengths.size // 2]

pdk = gf.get_active_pdk()

cross_section = pdk.get_cross_section("strip", width=width)

layer_stack = pdk.get_layer_stack()
core = layer_stack.layers["core"]
clad = layer_stack.layers["clad"]
box = layer_stack.layers["box"]

print(
    f"""Stack:
- {clad.material} clad with {clad.thickness}µm
- {core.material} clad with {core.thickness}µm
- {box.material} clad with {box.thickness}µm"""
)

# %% [markdown]
# The first component we need to design is the DC.  We model the coupling reagion first:
#


# %%
@gf.cell
def coupler_straight(
    gap: float, length: float, cross_section: gf.typings.CrossSectionSpec = "strip"
) -> gf.Component:
    cs = gf.get_cross_section(cross_section)
    Δy = (cs.width + gap) / 2

    sc = gf.components.straight(length, cross_section=cross_section)

    c = gf.Component()
    arm0 = c.add_ref(sc).movey(Δy)
    arm1 = c.add_ref(sc).movey(-Δy)

    c.add_port(name="o1", port=arm1.ports["o1"])
    c.add_port(name="o2", port=arm0.ports["o1"])
    c.add_port(name="o3", port=arm0.ports["o2"])
    c.add_port(name="o4", port=arm1.ports["o2"])

    return c


c = coupler_straight(gap, 2.0, cross_section=cross_section)
c.plot()

# %% [markdown]
# Next, we design the waveguide separating region using S bends.
#


# %%
@gf.cell
def coupler_splitter(
    gap: float,
    separation: float = 4.0,
    bend_factor: float = 3.0,
    cross_section: gf.typings.CrossSectionSpec = "strip",
) -> gf.Component:
    cs = gf.get_cross_section(cross_section)
    Δy = (cs.width + gap) / 2
    Δx = (separation / 2 - Δy) * bend_factor
    assert Δy < separation / 2

    sc = gf.components.bend_s(
        (Δx, separation / 2 - Δy), 199, cross_section=cross_section
    )

    c = gf.Component()
    arm0 = c.add_ref(sc).movey(Δy)
    arm1 = c.add_ref(sc).movey(Δy).mirror((0, 0), (1, 0))

    c.add_port(name="o1", port=arm1.ports["o1"])
    c.add_port(name="o2", port=arm0.ports["o1"])
    c.add_port(name="o3", port=arm0.ports["o2"])
    c.add_port(name="o4", port=arm1.ports["o2"])

    return c


separation = 2.0
bend_factor = 4.0

c = coupler_splitter(
    gap, separation=separation, bend_factor=bend_factor, cross_section=cross_section
)
c.plot()

# %% [markdown]
# To complete the DC, we join the previous designs in a full component.
#


# %%
@gf.cell
def coupler_symmetric(
    gap: float,
    length: float,
    separation: float = 4.0,
    bend_factor: float = 3.0,
    cross_section: gf.typings.CrossSectionSpec = "strip",
) -> gf.Component:
    splitter = coupler_splitter(
        gap, separation=separation, bend_factor=bend_factor, cross_section=cross_section
    )
    straight = coupler_straight(gap, length, cross_section=cross_section)

    x = splitter.ports["o3"].x

    c = gf.Component()
    sp0 = c.add_ref(splitter).mirror((x, 0), (x, 1))
    st = c.add_ref(straight)
    st.connect("o1", sp0.ports["o1"])
    sp1 = c.add_ref(splitter)
    sp1.connect("o1", st.ports["o4"])

    c.add_port(name="o1", port=sp0.ports["o4"])
    c.add_port(name="o2", port=sp0.ports["o3"])
    c.add_port(name="o3", port=sp1.ports["o3"])
    c.add_port(name="o4", port=sp1.ports["o4"])

    return c


c = coupler_symmetric(
    gap,
    2.0,
    separation=separation,
    bend_factor=bend_factor,
    cross_section=cross_section,
)
c.plot()

# %% [markdown]
# We use the `tidy3d` plugin to atumatically create an FDTD simulation of the complete DC.  We can inspect the simulation and port modes before running it to make sure our design is correct.

# %%
coupler = coupler_symmetric(
    gap,
    2.0,
    separation=separation,
    bend_factor=bend_factor,
    cross_section=cross_section,
)

sim_specs = dict(
    layer_stack=layer_stack,
    wavelength_start=wavelengths[0],
    wavelength_stop=wavelengths[-1],
    wavelength_points=wavelengths.size,
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=20),
)

simulation = gt.get_simulation(
    coupler,
    ymargin=2.0,
    num_modes=2,
    plot_modes=True,
    with_all_monitors=True,
    **sim_specs,
)

simulation.plot(z=0)
simulation.plot(x=0)

# %% [markdown]
# Because of the smooth S bend regions, the usual analytical models to calculate the power ratio of the DC give only a rough estimate.  We sweep a range of DC lengths based on those estimates to find the dimensions required in our design for the given PDK.

# %%
sim_lengths = np.linspace(0.0, 12.0, 13)

sims = gt.write_sparameters_batch(
    [
        {
            "component": coupler_symmetric(
                gap,
                length,
                separation=separation,
                bend_factor=bend_factor,
                cross_section=cross_section,
            ),
            "port_source_names": ["o1"],
            "ymargin": 2.0,
            "num_modes": 2,
            "filepath": PATH.sparameters_repo / f"dc_{length}",
        }
        for length in sim_lengths
    ],
    **sim_specs,
)

s_params_list = [sim.result() for sim in sims]

wavelengths = s_params_list[0]["wavelengths"]
drop = np.array([np.abs(s["o3@0,o1@0"]) ** 2 for s in s_params_list])
thru = np.array([np.abs(s["o4@0,o1@0"]) ** 2 for s in s_params_list])
loss = 1 - (drop + thru)
sim_ratios = drop / (drop + thru)

fig, ax = plt.subplots(2, 2, figsize=(12, 6))

for i in range(0, wavelengths.size, 20):
    ax[0, 0].plot(sim_lengths, drop[:, i], label=f"{wavelengths[i]}µm")

for i, length in enumerate(sim_lengths):
    ax[0, 1].plot(wavelengths, drop[i, :], label=f"{length}µm")
    ax[1, 0].plot(wavelengths, sim_ratios[i, :], label=f"{length}µm")
    ax[1, 1].plot(wavelengths, loss[i, :], label=f"{length}µm")

ax[0, 0].set_xlabel("Coupler length (µm)")
ax[0, 0].set_ylabel("Drop ratio")
ax[0, 1].set_xlabel("λ (µm)")
ax[0, 1].set_ylabel("Drop ratio")
ax[1, 0].set_xlabel("λ (µm)")
ax[1, 0].set_ylabel("Power ratio")
ax[1, 1].set_xlabel("λ (µm)")
ax[1, 1].set_ylabel("Loss")
ax[0, 0].legend()
fig.tight_layout()

# %% [markdown]
# Now we crete a fitting function to calculate the DC length for a given power ratio.
#
# In the filter specification, the desired ratios are 0.5, 0.13, 0.12, 0.5, and 0.25.  We calculate the DC lengths accordingly.
#


# %%
def coupler_length(λ: float = 1.55, power_ratio: float = 0.5):
    i0 = np.argmin(np.abs(wavelengths - λ))
    i1 = min(i0 + 1, len(wavelengths) - 1) if λ > wavelengths[i] else max(i0 - 1, 0)
    if i1 != i0:
        pr = (
            sim_ratios[:, i0] * (wavelengths[i1] - λ)
            + sim_ratios[:, i1] * (λ - wavelengths[i0])
        ) / (wavelengths[i1] - wavelengths[i0])
    else:
        pr = sim_ratios[:, i0]
    y = pr - power_ratio
    root_indices = np.flatnonzero(y[1:] * y[:-1] <= 0)
    if len(root_indices) == 0:
        return sim_lengths[np.argmin(np.abs(y))]
    j = root_indices[0]
    return (
        sim_lengths[j] * (pr[j + 1] - power_ratio)
        + sim_lengths[j + 1] * (power_ratio - pr[j])
    ) / (pr[j + 1] - pr[j])


power_ratios = [0.50, 0.13, 0.12, 0.50, 0.25]
lengths = [coupler_length(lda_c, pr) for pr in power_ratios]
print("Power ratios:", power_ratios)
print("Lengths:", lengths)

# %% [markdown]
# Finally, we simulate the DCs with the calculated lengths to guarantee the fitting error is within tolerance.  As expected, all DCs have the correct power ratios at the central wavelength.

# %%
sims = gt.write_sparameters_batch(
    [
        {
            "component": coupler_symmetric(
                gap,
                length,
                separation=separation,
                bend_factor=bend_factor,
                cross_section=cross_section,
            ),
            "ymargin": 2.0,
            "num_modes": 2,
            "port_source_names": ["o1"],
            "port_symmetries": {
                "o1@0,o1@0": {"o2@0,o2@0", "o3@0,o3@0", "o4@0,o4@0"},
                "o2@0,o1@0": {"o1@0,o2@0", "o4@0,o3@0", "o3@0,o4@0"},
                "o3@0,o1@0": {"o1@0,o3@0", "o4@0,o2@0", "o2@0,o4@0"},
                "o4@0,o1@0": {"o1@0,o4@0", "o3@0,o2@0", "o2@0,o3@0"},
            },
            "filepath": PATH.sparameters_repo / f"dc_{length}",
        }
        for length in lengths
    ],
    **sim_specs,
)

s_params_list = [sim.result() for sim in sims]

fig, ax = plt.subplots(1, 3, figsize=(12, 3))
errors = []
i = wavelengths.size // 2

for pr, sp in zip(power_ratios, s_params_list):
    drop = np.abs(sp["o3@0,o1@0"]) ** 2
    thru = np.abs(sp["o4@0,o1@0"]) ** 2

    assert lda_c == wavelengths[i]
    errors.append(drop[i] / (thru[i] + drop[i]) - pr)

    ax[0].plot(wavelengths, thru, label=f"{1 - pr}")
    ax[1].plot(wavelengths, drop, label=f"{pr}")
    ax[2].plot(wavelengths, 1 - thru - drop)

ax[0].set_ylabel("Thru ratio")
ax[1].set_ylabel("Drop ratio")
ax[2].set_ylabel("Loss")
ax[0].set_ylim(0, 1)
ax[1].set_ylim(0, 1)
ax[0].legend()
ax[1].legend()
fig.tight_layout()

print(errors)

# %% [markdown]
# Now we have to design the arms of each MZI.  The most important parameter here is their free spectral range (FSR), which comes from the path length difference and the group index of the waveguide at the central wavelength:
#
# $$\text{FSR} = \frac{\lambda_c^2}{n_g \Delta L}$$
#
# We calculate the group index for our waveguides through `tidy3d`'s local mode solver.  Because we're interested in precise dispersion, we use a dense mesh and high precision in these calculations.
#
# The path length differences for the MZIs are $\Delta L$,  $2\Delta L$, $L_\pi - 2\Delta L$, and $-2\Delta L$, with $L_\pi$ the length required for $\pi$ phase shift (negative values indicate a delay in the opposite arm to positive values).
#


# %%
def mzi_path_difference(waveguide: gt.modes.Waveguide, group_index: float, fsr: float):
    return waveguide.wavelength**2 / (fsr * group_index)


nm = 1e-3

mode_solver_specs = dict(
    core_material=core.material,
    clad_material=clad.material,
    core_width=width,
    core_thickness=core.thickness,
    box_thickness=min(2.0, box.thickness),
    clad_thickness=min(2.0, clad.thickness),
    side_margin=2.0,
    num_modes=2,
    grid_resolution=20,
    precision="double",
)

waveguide_solver = gt.modes.Waveguide(
    wavelength=lda_c, **mode_solver_specs, group_index_step=True
)

waveguide_solver.plot_field(field_name="Ex", mode_index=0)
ng = waveguide_solver.n_group[0]
ne = waveguide_solver.n_eff[0].real
print(f"ne = {ne}, ng = {ng}")

length_delta = mzi_path_difference(waveguide_solver, ng, fsr)
length_pi = lda_c / (2 * ne)
mzi_deltas = [
    length_delta,
    2 * length_delta,
    length_pi - 2 * length_delta,
    -2 * length_delta,
]
print(f"Path difference (ΔL = {length_delta}, Lπ = {length_pi}):", mzi_deltas)

# %% [markdown]
# Next we create a helper function that returns the MZI arms for a given length difference, respecting the bend radius defined in our PDK.
#


# %%
from typing import Tuple


def mzi_arms(
    mzi_delta: float,
    separation: float = 4.0,
    cross_section: gf.typings.CrossSectionSpec = "strip",
) -> Tuple[gf.ComponentReference, gf.ComponentReference]:
    bend = gf.components.bend_euler(cross_section=cross_section)

    if mzi_delta > 0:
        arm0 = [
            gf.ComponentReference(bend),
            gf.ComponentReference(
                gf.components.straight(mzi_delta / 2, cross_section=cross_section)
            ),
            gf.ComponentReference(bend).mirror(),
            gf.ComponentReference(
                gf.components.straight(separation * 2, cross_section=cross_section)
            ),
            gf.ComponentReference(bend).mirror(),
            gf.ComponentReference(
                gf.components.straight(mzi_delta / 2, cross_section=cross_section)
            ),
            gf.ComponentReference(bend),
        ]
        arm1 = [
            gf.ComponentReference(
                gf.components.straight(separation, cross_section=cross_section)
            ),
            gf.ComponentReference(bend),
            gf.ComponentReference(bend).mirror(),
            gf.ComponentReference(bend).mirror(),
            gf.ComponentReference(bend),
            gf.ComponentReference(
                gf.components.straight(separation, cross_section=cross_section)
            ),
        ]
    else:
        arm0 = [
            gf.ComponentReference(
                gf.components.straight(separation, cross_section=cross_section)
            ),
            gf.ComponentReference(bend).mirror(),
            gf.ComponentReference(bend),
            gf.ComponentReference(bend),
            gf.ComponentReference(bend).mirror(),
            gf.ComponentReference(
                gf.components.straight(separation, cross_section=cross_section)
            ),
        ]
        arm1 = [
            gf.ComponentReference(bend).mirror((0, 0), (1, 0)),
            gf.ComponentReference(
                gf.components.straight(-mzi_delta / 2, cross_section=cross_section)
            ),
            gf.ComponentReference(bend),
            gf.ComponentReference(
                gf.components.straight(separation * 2, cross_section=cross_section)
            ),
            gf.ComponentReference(bend),
            gf.ComponentReference(
                gf.components.straight(-mzi_delta / 2, cross_section=cross_section)
            ),
            gf.ComponentReference(bend).mirror(),
        ]

    return (arm0, arm1)


arm_references = mzi_arms(
    mzi_deltas[0], separation=separation, cross_section=cross_section
)

# %% [markdown]
# Now we can put all pieces together to layout the complete cascaded MZI filter:
#


# %%
@gf.cell
def cascaded_mzi(
    coupler_gaps,
    coupler_lengths,
    mzi_deltas,
    separation: float = 4.0,
    bend_factor: float = 3.0,
    cross_section: gf.typings.CrossSectionSpec = "strip",
) -> gf.Component:
    assert len(coupler_lengths) > 0
    assert len(coupler_gaps) == len(coupler_lengths)
    assert len(mzi_deltas) + 1 == len(coupler_lengths)
    c = gf.Component()

    coupler = c.add_ref(
        coupler_symmetric(
            coupler_gaps[0],
            coupler_lengths[0],
            separation=separation,
            bend_factor=bend_factor,
            cross_section=cross_section,
        )
    )
    c.add_port(name="o1", port=coupler.ports["o1"])
    c.add_port(name="o2", port=coupler.ports["o2"])

    for g, l, dl in zip(coupler_gaps[1:], coupler_lengths[1:], mzi_deltas):
        arm0, arm1 = mzi_arms(dl, separation=separation, cross_section=cross_section)
        c.add(arm0)
        c.add(arm1)
        arm0[0].connect("o1", coupler.ports["o3"])
        arm1[0].connect("o1", coupler.ports["o4"])
        for arm in (arm0, arm1):
            for r0, r1 in zip(arm[:-1], arm[1:]):
                r1.connect("o1", r0.ports["o2"])
        coupler = c.add_ref(
            coupler_symmetric(
                g,
                l,
                separation=separation,
                bend_factor=bend_factor,
                cross_section=cross_section,
            )
        )
        coupler.connect("o1", arm1[-1].ports["o2"])

    c.add_port(name="o3", port=coupler.ports["o3"])
    c.add_port(name="o4", port=coupler.ports["o4"])

    return c


layout = cascaded_mzi(
    coupler_gaps=[gap] * len(lengths),
    coupler_lengths=lengths,
    mzi_deltas=mzi_deltas,
    separation=separation,
    bend_factor=bend_factor,
    cross_section=cross_section,
)
layout.plot()

# %% [markdown]
# Finally, we want to build a complete simulation of the filter based on individual models for its components.
#
# We extract the filter netlist and verify we'll need models for the straight and bend sections, as well as for the DCs.

# %%
netlist = layout.get_netlist()
{v["component"] for v in netlist["instances"].values()}

# %% [markdown]
# The model for the straight sections is based directly on the waveguide mode, including dispersion effects.

# %%
straight_wavelengths = jnp.linspace(wavelengths[0], wavelengths[-1], 11)
straight_neffs = np.empty(straight_wavelengths.size, dtype=complex)

waveguide_solver = gt.modes.Waveguide(
    wavelength=list(straight_wavelengths), **mode_solver_specs
)
straight_neffs = waveguide_solver.n_eff[:, 0]

plt.plot(straight_wavelengths, straight_neffs.real, ".-")
plt.xlabel("λ (µm)")
plt.ylabel("n_eff")


# %%
@jax.jit
def straight_model(wl=1.55, length: float = 1.0):
    s21 = jnp.exp(
        2j * jnp.pi * jnp.interp(wl, straight_wavelengths, straight_neffs) * length / wl
    )
    zero = jnp.zeros_like(wl)
    return {
        ("o1", "o1"): zero,
        ("o1", "o2"): s21,
        ("o2", "o1"): s21,
        ("o2", "o2"): zero,
    }


straight_model()

# %% [markdown]
# For the bends, we want to include the full S matrix, because we are not using a circular shape, so simple modal decomposition becomes less accurate.  Similarly, we want to use the full simulated S matrix from the DCs in our model, instead of analytical approximations.
#
# We encapsulate the S parameter calculation in a helper function that generates the `jax` model for each component.
#


# %%
def bend_model(cross_section: gf.typings.CrossSectionSpec = "strip"):
    component = gf.components.bend_euler(cross_section=cross_section)
    s = gt.write_sparameters(
        component=component,
        num_modes=2,
        port_source_names=["o1"],
        filepath=PATH.sparameters_repo / f"{component.name}.npz",
        **sim_specs,
    )
    wavelengths = s.pop("wavelengths")

    @jax.jit
    def _model(wl=1.55):
        s11 = jnp.interp(wl, wavelengths, s["o1@0,o1@0"])
        s21 = jnp.interp(wl, wavelengths, s["o2@0,o1@0"])
        return {
            ("o1", "o1"): s11,
            ("o1", "o2"): s21,
            ("o2", "o1"): s21,
            ("o2", "o2"): s11,
        }

    return _model


bend_model(cross_section=cross_section)()

# %%
c = gf.Component(name="bend")
ref = c.add_ref(gf.components.bend_euler(cross_section=cross_section))
c.add_ports(ref.ports)
x, _ = sax.circuit(
    c.get_netlist(), {"bend_euler": bend_model(cross_section=cross_section)}
)

s = x(wl=wavelengths)
plt.plot(wavelengths, jnp.abs(s[("o1", "o2")]) ** 2)
plt.ylabel("S21")
plt.xlabel("λ (µm)")


# %%
def coupler_model(
    gap: float = 0.1,
    length: float = 1.0,
    separation: float = 4.0,
    bend_factor: float = 3.0,
    cross_section: gf.typings.CrossSectionSpec = "strip",
):
    component = coupler_symmetric(
        gap,
        length,
        separation=separation,
        bend_factor=bend_factor,
        cross_section=cross_section,
    )
    s = gt.write_sparameters(
        component=component,
        ymargin=2.0,
        num_modes=2,
        port_source_names=["o1"],
        filepath=PATH.sparameters_repo / f"{component.name}.npz",
        **sim_specs,
    )
    wavelengths = s.pop("wavelengths")

    @jax.jit
    def _model(wl=1.55):
        s11 = jnp.interp(wl, wavelengths, s["o1@0,o1@0"])
        s21 = jnp.interp(wl, wavelengths, s["o2@0,o1@0"])
        s31 = jnp.interp(wl, wavelengths, s["o3@0,o1@0"])
        s41 = jnp.interp(wl, wavelengths, s["o4@0,o1@0"])
        return {
            ("o1", "o1"): s11,
            ("o1", "o2"): s21,
            ("o1", "o3"): s31,
            ("o1", "o4"): s41,
            ("o2", "o1"): s21,
            ("o2", "o2"): s11,
            ("o2", "o3"): s41,
            ("o2", "o4"): s31,
            ("o3", "o1"): s31,
            ("o3", "o2"): s41,
            ("o3", "o3"): s11,
            ("o3", "o4"): s21,
            ("o4", "o1"): s41,
            ("o4", "o2"): s31,
            ("o4", "o3"): s21,
            ("o4", "o4"): s11,
        }

    return _model


coupler_model(
    gap,
    lengths[0],
    separation=separation,
    bend_factor=bend_factor,
    cross_section=cross_section,
)()

# %% [markdown]
# We must take care of using one model for each DC based on its length, so we use another helper function that iterates over the netlist instances and generates the appropriate model for each one:
#


# %%
def patch_netlist(netlist, models, models_to_patch):
    instances = netlist["instances"]
    for name in instances:
        model = instances[name]
        if model["component"] in models_to_patch:
            component = model["component"]
            i = 0
            new_component = f"{component}_v{i}"
            while new_component in models:
                i += 1
                new_component = f"{component}_v{i}"
            models[new_component] = models_to_patch[model["component"]](
                **model["settings"]
            )
            del model["settings"]
            model["component"] = new_component
    return netlist, models


pl_set = sorted(set(zip(power_ratios, lengths)))
fig, ax = plt.subplots(len(pl_set), 1, figsize=(4, 3 * len(pl_set)))

for i, (pr, l) in enumerate(pl_set):
    c = gf.Component(name="single mzi 2")
    ref = c.add_ref(
        coupler_symmetric(
            gap,
            l,
            separation=separation,
            bend_factor=bend_factor,
            cross_section=cross_section,
        )
    )
    c.add_ports(ref.ports)
    netlist, models = patch_netlist(
        c.get_netlist(), {}, {"coupler_symmetric": coupler_model}
    )
    x, _ = sax.circuit(netlist, models)

    s = x(wl=wavelengths)
    ax[i].plot(wavelengths, jnp.abs(s[("o1", "o3")]) ** 2, label="Cross")
    ax[i].plot(wavelengths, jnp.abs(s[("o1", "o4")]) ** 2, label="Through")
    ax[i].axvline(lda_c, c="tab:gray", ls=":", lw=1)
    ax[i].set_ylim(0, 1)
    ax[i].set_xlabel("λ (µm)")
    ax[i].set_title(f"l = {l:.2f} µm ({pr})")

ax[0].legend()
fig.tight_layout()

# %% [markdown]
# Finally, we can simulate the complete filter response around the central wavelength and get the desired FSR and box-like shape.

# %%
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

layout = cascaded_mzi(
    coupler_gaps=[gap] * len(lengths),
    coupler_lengths=lengths,
    mzi_deltas=mzi_deltas,
    separation=separation,
    bend_factor=bend_factor,
    cross_section=cross_section,
)
netlist, models = patch_netlist(
    layout.get_netlist(),
    {"straight": straight_model, "bend_euler": bend_model(cross_section=cross_section)},
    {"coupler_symmetric": coupler_model},
)
circuit, _ = sax.circuit(netlist, models)

lda = np.linspace(1.5, 1.6, 1001)
s = circuit(wl=lda)
ax.plot(lda, 20 * jnp.log10(jnp.abs(s[("o1", "o3")])), label="Cross")
ax.plot(lda, 20 * jnp.log10(jnp.abs(s[("o1", "o4")])), label="Thru")
ax.set_ylim(-30, 0)
ax.set_xlabel("λ (µm)")
ax.legend()
