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
# ## Design for testing
#
# To measure your chips after fabrication you need to decide your test configurations. This includes things like:
#
# - `Individual input and output fibers` versus `fiber array`. You can use `add_fiber_array` for easier testing and higher throughput, or `add_fiber_single` for the flexibility of single fibers.
# - Fiber array pitch (127um or 250um) if using a fiber array.
# - Pad pitch for DC and RF high speed probes (100, 125, 150, 200um). Probe configuration (GSG, GS ...)
# - Test layout for DC, RF and optical fibers.
#
#
# To enable automatic testing you can add labels the devices that you want to test. GDS labels are not fabricated and are only visible in the GDS file.
#
# Lets review some different automatic labeling schemas:
#
# 1. One label per test site (Component) that includes settings, electrical ports and optical ports.
# 2. SiEPIC labels , labels only one of the grating couplers from the fiber array.
# 3. EHVA automatic testers, include a Label component declaration as described in this [doc](https://drive.google.com/file/d/1kbQNrVLzPbefh3by7g2s865bcsA2vl5l/view)
#
#
# Most gdsfactory examples add south grating couplers on the south and RF or DC signals to the north. However if you need RF and DC pads, you have to make sure RF pads are orthogonal to the DC Pads. For example, you can use EAST/WEST for RF and NORTH for DC.

# %%
from functools import partial

import ipywidgets
from IPython.display import display
from omegaconf import OmegaConf

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.labels import add_label_ehva


gf.CONF.display_type = "klayout"
gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# %% [markdown]
# ### 1. Test Sites Labels
#
# Each test site labels includes:
#
# - Optical and electrical ports
# - Component settings
# - test and data analysis information. Such as Design of Experiment (DOE) id.

# %%
c = gf.components.mzi_phase_shifter()
c = gf.components.add_fiber_array_optical_south_electrical_north(
    c,
    test_info=dict(
        doe="mzis",
        data_analysis="mzi_phase_shifter",
        test_sequence="optical_electrial",
    ),
)
c.plot()

# %%
c.labels

# %%
import json

json.loads(c.labels[0].text)

# %% [markdown]
# ### 2. SiEPIC labels
#
# Labels follow format `opt_in_{polarization}_{wavelength}_device_{username}_({component_name})-{gc_index}-{port.name}` and you only need to label the input port of the fibe array.
# This also includes one label per test site.

# %%
mmi = gf.components.mmi2x2()
mmi_te_siepic = gf.labels.add_fiber_array_siepic(component=mmi)
mmi_te_siepic.show()
mmi_te_siepic.plot()

# %%
mmi_te_siepic.ports

# %%
labels = mmi_te_siepic.get_labels()

for label in labels:
    print(label.text)

# %% [markdown]
# ### 3. EHVA labels

# %%
add_label_ehva_demo = partial(add_label_ehva, die="demo_die")
mmi = gf.c.mmi2x2(length_mmi=2.2)
mmi_te_ehva = gf.routing.add_fiber_array(
    mmi, get_input_labels_function=None, decorator=add_label_ehva_demo
)
mmi_te_ehva.show()
mmi_te_ehva.plot()

# %%
labels = mmi_te_ehva.get_labels(depth=0)

for label in labels:
    print(label.text)

# %% [markdown]
# One advantage of the EHVA formats is that you can track any changes on the components directly from the GDS label, as the label already stores any changes of the child device, as well as any settings that you specify.
#
# Settings can have many levels of hierarchy, but you can still access any children setting with `:` notation.
#
# ```
# grating_coupler:
#     function: grating_coupler_elliptical_trenches
#     settings:
#         polarization: te
#         taper_angle: 35
#
# ```

# %%
add_label_ehva_demo = partial(
    add_label_ehva,
    die="demo_die",
    metadata_include_parent=["grating_coupler:settings:polarization"],
)
mmi = gf.components.mmi2x2(length_mmi=10)
mmi_te_ehva = gf.routing.add_fiber_array(
    mmi, get_input_labels_function=None, decorator=add_label_ehva_demo
)
mmi_te_ehva.show()
mmi_te_ehva.plot()

# %%
labels = mmi_te_ehva.get_labels(depth=0)

for label in labels:
    print(label.text)

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
# Then we add spirals with different lengths to measure waveguide propagation loss.

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
c.show()
c.plot()

# %% [markdown]
# You can also add some physical labels that will be fabricated.
# For example you can add prefix `S` at the `north-center` of each spiral using `text_rectangular` which is DRC clean and anchored on `nc` (north-center)

# %%
text_metal3 = partial(gf.components.text_rectangular_multi_layer, layers=((49, 0),))

m = gf.pack(sweep, text=text_metal3, text_anchors=("nc",), text_prefix="s")
c = m[0]
c.plot()

# %%
text_metal2 = partial(gf.components.text, layer=(45, 0))

m = gf.pack(sweep, text=text_metal2, text_anchors=("nc",), text_prefix="s")
c = m[0]
c.plot()

# %% [markdown]
# ## Grid
#
# You can also pack components with a constant spacing.

# %%
g = gf.grid(sweep)
g.plot()

# %%
gh = gf.grid(sweep, shape=(1, len(sweep)))
gh.plot()

# %%
gh_ymin = gf.grid(sweep, shape=(1, len(sweep)), align_y="ymin")
gh_ymin.plot()

# %% [markdown]
# You can also add text labels to each element of the sweep

# %%
gh_ymin = gf.grid_with_text(
    sweep, shape=(1, len(sweep)), align_y="ymin", text=text_metal3
)
gh_ymin.plot()

# %% [markdown]
# You can modify the text by customizing the `text_function` that you pass to `grid_with_text`

# %%
gh_ymin_m2 = gf.grid_with_text(
    sweep, shape=(1, len(sweep)), align_y="ymin", text=text_metal2
)
gh_ymin_m2.plot()

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


c = mask(cache=False)
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
# ## Metadata
#
# When saving GDS files is also convenient to store the metadata settings that you used to generate the GDS file.
#
# We recommend storing all the device metadata in GDS labels but you can also store it in a separate YAML file.
#
# ### Metadata in separate YAML file (not recommended)

# %%
import gdsfactory as gf


@gf.cell
def wg():
    c = gf.Component()
    c.info["doe"] = ["rings", 1550, "te", "phase_shifter"]
    c.info["test_sequence"] = ["optical", "electrical_sweep"]
    c.info["data_analysis"] = [
        "remove_baseline",
        "extract_fsr",
        "extract_loss",
        "extract_power_per_pi",
    ]
    return c


c = wg()
c.pprint()
gdspath = c.write_gds("demo.gds", with_metadata=True)


# %% [markdown]
# ### Metadata in the GDS file
#
# You can use GDS labels to store device information such as settings and port locations.
#
# The advantage of GDS labels is that they are all stored in the same file.
#
# We define a single label for each test site, and the label contains all the testing and data analysis information.

# %%
test_info_mzi_heaters = dict(
    doe="mzis_heaters",
    analysis="mzi_heater_phase_shifter_length",
    measurement="optical_mzi_heater",
)
test_info_ring_heaters = dict(
    doe="ring_heaters_coupling_length",
    analysis="ring_heater",
    measurement="optical_ring_heater",
)

mzis = [
    gf.components.mzi2x2_2x2_phase_shifter(length_x=lengths)
    for lengths in [100, 200, 300]
]

rings = [
    gf.components.ring_single_heater(length_x=length_x) for length_x in [10, 20, 30]
]

mzis_te = [
    gf.components.add_fiber_array_optical_south_electrical_north(
        mzi,
        electrical_port_names=["top_l_e2", "top_r_e2"],
        **test_info_mzi_heaters,
    )
    for mzi in mzis
]
rings_te = [
    gf.components.add_fiber_array_optical_south_electrical_north(
        ring, electrical_port_names=["l_e2", "r_e2"], **test_info_ring_heaters
    )
    for ring in rings
]
c = gf.pack(mzis_te + rings_te)[0]
c.plot()

# %% [markdown]
# ## Test manifest
#
# Each site has a test label in JSON with all the settings.

# %%
import pandas as pd

gdspath = c.write_gds()
csvpath = gf.labels.write_labels.write_labels_gdstk(
    gdspath, debug=True, prefixes=["{"], layer_label="TEXT"
)
df = pd.read_csv(csvpath)
df

# %% [markdown]
# As you can see there are 6 devices, each of which has optical and electrical ports.
#
# You can turn each label into a test manifest CSV file, which will be used to interface with your lab instrumentation.
#
# Having a general case for interfacing with test equipment is pretty hard, so this is just a guideline on how this could be done.
#
# Each `doe` Design of Experiment (DOE) will have a different output, can run a different `measurement` and have different `analysis` function code and settings, which would have to be defined separately.
#
# This would require having a separate `measurement_settings.yml` and  `analysis_settings.yml`  with all the settings in YAML format

# %%
from gdsfactory.labels.write_test_manifest import write_test_manifest

dm = write_test_manifest(csvpath)
dm

# %% [markdown]
# ![](https://i.imgur.com/FnOCb6H.png)

# %% [markdown]
#
