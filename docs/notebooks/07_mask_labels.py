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
# # Die assembly with labels (deprecated)
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
# ## Automated testing using labels
#
# This is deprecated, we recommend exposing all ports and writing the test manifest directly.
# However you can also do automatic testing by adding labels the devices that you want to test.
# GDS labels are not fabricated and are only visible in the GDS file.
#
# Lets review some different automatic labeling schemas:
#
# 1. One label per test alignment that includes settings, electrical ports and optical ports.
# 2. SiEPIC labels: only the laser input grating coupler from the fiber array has a label, which is the second port from left to right.
# 3. EHVA automatic testers, include a Label component declaration as described in this [doc](https://drive.google.com/file/d/1kbQNrVLzPbefh3by7g2s865bcsA2vl5l/view)
#
# Most gdsfactory examples add south grating couplers on the south and RF or DC signals to the north. However if you need RF and DC pads, you have to make sure RF pads are orthogonal to the DC Pads. For example, you can use EAST/WEST for RF and NORTH for DC.
#
#
# You can also use code in `gf.labels.write_labels` to store the labels into CSV and `gf.labels.write_test_manifest`
#
# ### 1. Test Sites Labels
#
# Each alignment site includes a label with the measurement and analysis settings:
#
# - Optical and electrical port locations for each alignment.
# - measurement settings.
# - Component settings for the analysis and test and data analysis information. Such as Design of Experiment (DOE) id.
#
#
# The default settings can be stored in a separate [CSV file](https://docs.google.com/spreadsheets/d/1845m-XZM8tZ1tNd8GIvAaq7ZE-iha00XNWa0XrEOabc/edit#gid=0)

# %%
info = dict(
    doe="mzis",
    analysis="mzi_phase_shifter",
    measurement="optical_loopback2_heater_sweep",
    measurement_settings=dict(v_max=5),
)

c = gf.components.mzi_phase_shifter()
c = gf.components.add_fiber_array_optical_south_electrical_north(
    c, info=info, decorator=add_label_json
)
c.plot()

# %%
c.labels

# %%
json.loads(c.labels[0].text)

# %%
c = gf.components.spiral_inner_io_fiber_array(
    length=20e3,
    decorator=gf.labels.add_label_json,
    info=dict(
        measurement="optical_loopback2",
        doe="spiral_sc",
        measurement_settings=dict(wavelength_alignment=1560),
    ),
)
c.plot()

# %%
json.loads(c.labels[0].text)

# %% [markdown]
# ### 2. SiEPIC labels
#
# Labels follow format `opt_in_{polarization}_{wavelength}_device_{username}_({component_name})-{gc_index}-{port.name}` and you only need to label the laser input port of the fiber array.
# This also includes one label per test site.

# %%
mmi = gf.components.mmi2x2()
mmi_te_siepic = gf.labels.add_fiber_array_siepic(component=mmi)
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
mmi_te_ehva.plot()

# %%
labels = mmi_te_ehva.get_labels(depth=0)

for label in labels:
    print(label.text)

# %% [markdown]
# ## Metadata
#
# When saving GDS files is also convenient to store the metadata settings that you used to generate the GDS file.
#
# We recommend storing all the device metadata in GDS labels but you can also store it in a separate YAML file.
#
# ### Metadata in separate YAML file

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
# We define a single label for each test site (Device Under Test), and the label contains all the measurement and data analysis information.

# %%
test_info_mzi_heaters = dict(
    doe="mzis_heaters",
    analysis="mzi_heater_phase_shifter_length",
    measurement="optical_loopback4_heater_sweep",
)
test_info_ring_heaters = dict(
    doe="ring_heaters_coupling_length",
    analysis="ring_heater",
    measurement="optical_loopback2_heater_sweep",
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
        info=test_info_mzi_heaters,
        decorator=gf.labels.add_label_json,
    )
    for mzi in mzis
]
rings_te = [
    gf.components.add_fiber_array_optical_south_electrical_north(
        ring,
        electrical_port_names=["l_e2", "r_e2"],
        info=test_info_ring_heaters,
        decorator=gf.labels.add_label_json,
    )
    for ring in rings
]
c = gf.pack(mzis_te + rings_te)[0]
c.show()
c.plot()

# %% [markdown]
# ## Test manifest from labels
#
# Each Device Under Test (test site) has a JSON test label with all the settings.
#
# You can define a [Test manifest](https://docs.google.com/spreadsheets/d/1845m-XZM8tZ1tNd8GIvAaq7ZE-iha00XNWa0XrEOabc/edit#gid=0) (also known as Test sequence) in CSV automatically from the labels.

# %%
import pandas as pd

gdspath = c.write_gds()
csvpath = gf.labels.write_labels.write_labels_gdstk(
    gdspath, debug=True, prefixes=["{"], layer_label="TEXT"
)
df = pd.read_csv(csvpath)
df

# %% [markdown]
# As you can see there are 6 devices with optical and electrical ports.
#
# You can turn each label into a test manifest CSV file to interface with your lab instrumentation functions.
#
# Each measurement will use a different `measurement` procedure and settings `measurement_settings`
#
# The default measurement settings for each functions can also be defined in a separate [CSV file](https://docs.google.com/spreadsheets/d/1845m-XZM8tZ1tNd8GIvAaq7ZE-iha00XNWa0XrEOabc/edit#gid=138229318) and easily editable with Excel or LibreOffice.

# %%
from gdsfactory.labels.write_test_manifest import write_test_manifest

dm = write_test_manifest(csvpath)
dm
