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
# # Die assembly
#
# With gdsfactory you can easily go from a simple Component, to a Component with many components inside.
#
# In the same way that you need to Layout for DRC (Design Rule Check) clean devices, you have to layout obeying the Design for Test (DFT) and Design for Packaging rules.
#
# ## Design for test
#
# To measure your chips after fabrication you need to decide your test configurations. This includes Design For Testing Rules like:
#
# - `Individual input and output fibers` versus `fiber array`. You can use `add_fiber_array` for easier testing and higher throughput, or `add_fiber_single` for the flexibility of single fibers.
# - Fiber array pitch (127um or 250um) if using a fiber array.
# - Pad pitch for DC and RF high speed probes (100, 125, 150, 200um). Probe configuration (GSG, GS ...)
# - Test layout for DC, RF and optical fibers.
#

# %%
from functools import partial

import json
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.labels import add_label_ehva, add_label_json


gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# %% [markdown]
# ## Pack
#
# Lets start with a resistance sweep, where you change the resistance width to measure sheet resistance.

# %%
sweep = [gf.components.resistance_sheet(width=width) for width in [1, 10, 100]]
m = gf.pack(sweep)
c = m[0]
c.plot()

# %% [markdown]
# Then we add spirals with different lengths to measure waveguide propagation loss. You can use both fiber array or single fiber.

# %%
from toolz import compose
from functools import partial
import gdsfactory as gf

c = gf.components.spiral_inner_io_fiber_array(
    length=20e3,
    decorator=gf.labels.add_label_json,
    info=dict(measurement="optical_loopback2"),
)
c.plot()

# %%
spiral = gf.components.spiral_inner_io_fiber_single()
spiral.plot()

# %%
spiral_te = gf.routing.add_fiber_single(
    gf.functions.rotate(gf.components.spiral_inner_io_fiber_single, 90)
)
spiral_te.plot()

# %%
# which is equivalent to
spiral_te = gf.compose(
    gf.routing.add_fiber_single,
    gf.functions.rotate90,
    gf.components.spiral_inner_io_fiber_single,
)
c = spiral_te(length=10e3)
c.plot()

# %%
add_fiber_single_no_labels = partial(
    gf.routing.add_fiber_single,
    get_input_label_text_function=None,
)

spiral_te = gf.compose(
    add_fiber_single_no_labels,
    gf.functions.rotate90,
    gf.components.spiral_inner_io_fiber_single,
)
sweep = [spiral_te(length=length) for length in [10e3, 20e3, 30e3]]
m = gf.pack(sweep)
c = m[0]
c.plot()

# %%
from toolz import compose
from functools import partial
import gdsfactory as gf

c = gf.components.spiral_inner_io_fiber_array(
    length=20e3,
    decorator=gf.labels.add_label_json,
    info=dict(measurement="optical_loopback2"),
)
c.show()
c.plot()

# %%
sweep = [
    gf.components.spiral_inner_io_fiber_array(
        length=length,
        decorator=gf.labels.add_label_json,
        info=dict(measurement="optical_loopback2"),
    )
    for length in [20e3, 30e3, 40e3]
]
m = gf.pack(sweep)
c = m[0]
c.show()
c.plot()

# %% [markdown]
# You can also add some physical labels that will be fabricated.
# For example you can add prefix `S` at the `north-center` of each spiral using `text_rectangular` which is DRC clean and anchored on `nc` (north-center)

# %%
text_metal = partial(gf.components.text_rectangular_multi_layer, layers=("M1",))

m = gf.pack(sweep, text=text_metal, text_anchors=("cw",), text_prefix="s")
c = m[0]
c.show()
c.plot()

# %% [markdown]
# ## Grid
#
# You can also pack components with a constant spacing.

# %%
g = gf.grid_with_component_name(sweep)
g.plot()

# %%
gh = gf.grid_with_component_name(sweep, shape=(1, len(sweep)))
gh.plot()

# %%
gh_ymin = gf.grid_with_component_name(sweep, shape=(len(sweep), 1), align_x="xmin")
gh_ymin.plot()

# %% [markdown]
# You can also add text labels to each element of the sweep

# %%
gh_ymin = gf.grid_with_text(
    sweep, shape=(len(sweep), 1), align_x="xmax", text=text_metal
)
gh_ymin.plot()

# %% [markdown]
# You have 2 ways of defining a mask:
#
# 1. in python
# 2. in YAML
#
#
# ## 1. Component in python
#
# You can define a Component top cell reticle or die using `grid` and `pack` python functions.

# %%
text_metal3 = partial(gf.components.text_rectangular_multi_layer, layers=((49, 0),))
grid = partial(gf.grid_with_text, text=text_metal3)
pack = partial(gf.pack, text=text_metal3)

gratings_sweep = [
    gf.components.grating_coupler_elliptical(taper_angle=taper_angle)
    for taper_angle in [20, 30, 40]
]
gratings = grid(gratings_sweep, text=None)
gratings.plot()

# %%
gratings_sweep = [
    gf.components.grating_coupler_elliptical(taper_angle=taper_angle)
    for taper_angle in [20, 30, 40]
]
gratings_loss_sweep = [
    gf.components.grating_coupler_loss_fiber_single(grating_coupler=grating)
    for grating in gratings_sweep
]
gratings = grid(
    gratings_loss_sweep, shape=(1, len(gratings_loss_sweep)), spacing=(40, 0)
)
gratings.plot()

# %%
sweep_resistance = [
    gf.components.resistance_sheet(width=width) for width in [1, 10, 100]
]
resistance = gf.pack(sweep_resistance)[0]
resistance.plot()

# %%
spiral_te = gf.compose(
    gf.routing.add_fiber_single,
    gf.functions.rotate90,
    gf.components.spiral_inner_io_fiber_single,
)
sweep_spirals = [spiral_te(length=length) for length in [10e3, 20e3, 30e3]]
spirals = gf.pack(sweep_spirals)[0]
spirals.plot()

# %%
mask = gf.pack([spirals, resistance, gratings])[0]
mask.plot()


# %% [markdown]
# As you can see you can define your mask in a single line.
#
# For more complex mask, you can also create a new cell to build up more complexity
#


# %%
@gf.cell
def mask():
    c = gf.Component()
    c << gf.pack([spirals, resistance, gratings])[0]
    c << gf.components.seal_ring(c.bbox)
    return c


c = mask()
c.plot()

# %% [markdown]
# ## 2. Component in YAML
#
# You can also define your component in YAML format thanks to `gdsfactory.read.from_yaml`
#
# You need to define:
#
# - instances
# - placements
# - routes (optional)
#
# and you can leverage:
#
# 1. `pack_doe`
# 2. `pack_doe_grid`

# %% [markdown]
# ### 2.1 pack_doe
#
# `pack_doe` places components as compact as possible.

# %%

c = gf.read.from_yaml(
    """
name: mask_grid

instances:
  rings:
    component: pack_doe
    settings:
      doe: ring_single
      settings:
        radius: [30, 50, 20, 40]
        length_x: [1, 2, 3]
      do_permutations: True
      function:
        function: add_fiber_array
        settings:
            fanout_length: 200

  mzis:
    component: pack_doe
    settings:
      doe: mzi
      settings:
        delta_length: [10, 100]
      function: add_fiber_array

placements:
  rings:
    xmin: 50

  mzis:
    xmin: rings,east
"""
)

c.plot()

# %% [markdown]
# ### 2.2 pack_doe_grid
#
# `pack_doe_grid` places each component on a regular grid

# %%
c = gf.read.from_yaml(
    """
name: mask_compact

instances:
  rings:
    component: pack_doe
    settings:
      doe: ring_single
      settings:
        radius: [30, 50, 20, 40]
        length_x: [1, 2, 3]
      do_permutations: True
      function:
        function: add_fiber_array
        settings:
            fanout_length: 200


  mzis:
    component: pack_doe_grid
    settings:
      doe: mzi
      settings:
        delta_length: [10, 100]
      do_permutations: True
      spacing: [10, 10]
      function: add_fiber_array

placements:
  rings:
    xmin: 50

  mzis:
    xmin: rings,east
"""
)
c.plot()


# %% [markdown]
# ## Automated testing exposing all ports
#
# You can promote all the ports that need to be tested to the top level component and then write a CSV test manifest.
#
# This is the recommended way for measuring components that have electrical and optical port.


# %%
def sample_reticle() -> gf.Component:
    """Returns MZI with TE grating couplers."""
    test_info_mzi_heaters = dict(
        doe="mzis_heaters",
        analysis="mzi_heater",
        measurement="optical_loopback4_heater_sweep",
    )
    test_info_ring_heaters = dict(
        doe="ring_heaters",
        analysis="ring_heater",
        measurement="optical_loopback2_heater_sweep",
    )

    mzis = [
        gf.components.mzi2x2_2x2_phase_shifter(
            length_x=length, name=f"mzi_heater_{length}"
        )
        for length in [100, 200, 300]
    ]
    rings = [
        gf.components.ring_single_heater(
            length_x=length_x, name=f"ring_single_heater_{length_x}"
        )
        for length_x in [10, 20, 30]
    ]

    spirals_sc = [
        gf.components.spiral_inner_io_fiber_array(
            name=f"spiral_sc_{int(length/1e3)}mm",
            length=length,
            info=dict(
                doe="spirals_sc",
                measurement="optical_loopback4",
                analysis="optical_loopback4_spirals",
            ),
        )
        for length in [20e3, 40e3, 60e3]
    ]

    mzis_te = [
        gf.components.add_fiber_array_optical_south_electrical_north(
            mzi,
            electrical_port_names=["top_l_e2", "top_r_e2"],
            info=test_info_mzi_heaters,
            name=f"{mzi.name}_te",
        )
        for mzi in mzis
    ]
    rings_te = [
        gf.components.add_fiber_array_optical_south_electrical_north(
            ring,
            electrical_port_names=["l_e2", "r_e2"],
            info=test_info_ring_heaters,
            name=f"{ring.name}_te",
        )
        for ring in rings
    ]

    components = mzis_te + rings_te + spirals_sc

    c = gf.pack(components)
    if len(c) > 1:
        raise ValueError(f"failed to pack into single group. Made {len(c)} groups.")
    return c[0]


c = sample_reticle()
c.plot()

# %%
c.pprint_ports()

# %%
df = gf.labels.get_test_manifest(c)
df

# %%
df.to_csv("test_manifest.csv")


# %%
def sample_reticle_grid() -> gf.Component:
    """Returns MZI with TE grating couplers."""
    test_info_mzi_heaters = dict(
        doe="mzis_heaters",
        analysis="mzi_heater",
        measurement="optical_loopback4_heater_sweep",
    )
    test_info_ring_heaters = dict(
        doe="ring_heaters",
        analysis="ring_heater",
        measurement="optical_loopback2_heater_sweep",
    )

    mzis = [
        gf.components.mzi2x2_2x2_phase_shifter(
            length_x=length, name=f"mzi_heater_{length}"
        )
        for length in [100, 200, 300]
    ]
    rings = [
        gf.components.ring_single_heater(
            length_x=length_x, name=f"ring_single_heater_{length_x}"
        )
        for length_x in [10, 20, 30]
    ]

    spirals_sc = [
        gf.components.spiral_inner_io_fiber_array(
            name=f"spiral_sc_{int(length/1e3)}mm",
            length=length,
            info=dict(
                doe="spirals_sc",
                measurement="optical_loopback4",
                analysis="optical_loopback4_spirals",
            ),
        )
        for length in [20e3, 40e3, 60e3]
    ]

    mzis_te = [
        gf.components.add_fiber_array_optical_south_electrical_north(
            mzi,
            electrical_port_names=["top_l_e2", "top_r_e2"],
            info=test_info_mzi_heaters,
            name=f"{mzi.name}_te",
        )
        for mzi in mzis
    ]
    rings_te = [
        gf.components.add_fiber_array_optical_south_electrical_north(
            ring,
            electrical_port_names=["l_e2", "r_e2"],
            info=test_info_ring_heaters,
            name=f"{ring.name}_te",
        )
        for ring in rings
    ]

    components = mzis_te + rings_te + spirals_sc

    return gf.grid_with_component_name(components)


c = sample_reticle_grid()
c.plot()

# %%
df = gf.labels.get_test_manifest(c)
df

# %%
df.to_csv("test_manifest.csv")

# %% [markdown]
# You can see a test manifest example [here](https://docs.google.com/spreadsheets/d/1845m-XZM8tZ1tNd8GIvAaq7ZE-iha00XNWa0XrEOabc/edit#gid=233591479)
