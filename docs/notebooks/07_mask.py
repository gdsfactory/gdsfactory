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
# # Die assembly
#
# With gdsfactory you can easily go from a simple Component, to a Component with many references/instances.
#
# ## Design for testing
#
# To measure your reticle / die after fabrication you need to decide your test configurations. This includes things like:
#
# - `Individual input and output fibers` versus `fiber array`. We recommend `fiber array` for easier testing and higher throughtput, but also understand the flexibility of single fibers.
# - Fiber array pitch (127um or 250um) if using a fiber array.
# - Pad pitch for DC and RF high speed probes (100, 125, 150, 200um). Probe configuration (GSG, GS ...)
# - Test layout for DC, RF and optical fibers.
#
#
# To enable automatic testing you can add labels the devices that you want to test. GDS labels are not fabricated and are only visible in the GDS file.
#
# Lets review some different automatic labeling schemas:
#
# 1. SiEPIC ubc Ebeam PDK schema, labels only one of the grating couplers from the fiber array.
# 2. EHVA automatic testers, include a Label component declaration as described in this [doc](https://drive.google.com/file/d/1kbQNrVLzPbefh3by7g2s865bcsA2vl5l/view)
# 3. Label all ports

# %% [markdown]
# ### 1. SiEPIC labels
#
# Labels follow format `opt_in_{polarization}_{wavelength}_device_{username}_({component_name})-{gc_index}-{port.name}`

# %% tags=[]
import ipywidgets
from IPython.display import display
from omegaconf import OmegaConf

from gdsfactory.labels import add_label_ehva, add_label_yaml
from gdsfactory.read.labels import add_port_markers
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# %% tags=[]
mmi = gf.components.mmi2x2()
mmi_te_siepic = gf.labels.add_fiber_array_siepic(component=mmi)
mmi_te_siepic

# %% tags=[]
mmi_te_siepic.ports

# %% tags=[]
labels = mmi_te_siepic.get_labels()

for label in labels:
    print(label.text)

# %% [markdown]
# ### 2. EHVA labels

# %% tags=[]
add_label_ehva_demo = gf.partial(add_label_ehva, die="demo_die")
mmi = gf.c.mmi2x2(length_mmi=2.2)
mmi_te_ehva = gf.routing.add_fiber_array(
    mmi, get_input_labels_function=None, decorator=add_label_ehva_demo
)
mmi_te_ehva

# %% tags=[]
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

# %% tags=[]
add_label_ehva_demo = gf.partial(
    add_label_ehva,
    die="demo_die",
    metadata_include_parent=["grating_coupler:settings:polarization"],
)
mmi = gf.components.mmi2x2(length_mmi=10)
mmi_te_ehva = gf.routing.add_fiber_array(
    mmi, get_input_labels_function=None, decorator=add_label_ehva_demo
)
mmi_te_ehva

# %% tags=[]
labels = mmi_te_ehva.get_labels(depth=0)

for label in labels:
    print(label.text)

# %% [markdown]
# ### 3. Dash separated labels
#
# You can also use labels with `GratingName-ComponentName-PortName`

# %%
from gdsfactory.routing.get_input_labels import get_input_labels_dash

c1 = gf.components.mmi1x2()
c2 = gf.routing.add_fiber_array(c1, get_input_labels_function=get_input_labels_dash)
c2.show()

# %% [markdown]
# ## Pack
#
# Lets start with a resistance sweep, where you change the resistance width to measure sheet resistance.

# %% tags=[]
sweep = [gf.components.resistance_sheet(width=width) for width in [1, 10, 100]]
m = gf.pack(sweep)
c = m[0]
c

# %% [markdown]
# Then we add spirals with different lengths to measure waveguide propagation loss.

# %% tags=[]
spiral = gf.components.spiral_inner_io_fiber_single()
spiral

# %% tags=[]
spiral_te = gf.routing.add_fiber_single(
    gf.functions.rotate(gf.components.spiral_inner_io_fiber_single, 90)
)
spiral_te

# %% tags=[]
# which is equivalent to
spiral_te = gf.compose(
    gf.routing.add_fiber_single,
    gf.functions.rotate90,
    gf.components.spiral_inner_io_fiber_single,
)
c = spiral_te(length=10e3)
c

# %% tags=[]
add_label_ehva_mpw1 = gf.partial(gf.labels.add_label_ehva, die="mpw1")
add_fiber_single_no_labels = gf.partial(
    gf.routing.add_fiber_single,
    get_input_label_text_function=None,
    decorator=add_label_ehva_mpw1,
)

spiral_te = gf.compose(
    add_fiber_single_no_labels,
    gf.functions.rotate90,
    gf.components.spiral_inner_io_fiber_single,
)
sweep = [spiral_te(length=length) for length in [10e3, 20e3, 30e3]]
m = gf.pack(sweep)
c = m[0]
c

# %% [markdown]
# Together with GDS labels that are not fabricated, you can also add some physical labels that will be fabricated.
#
# For example you can add prefix `S` at the `north-center` of each spiral using `text_rectangular` which is DRC clean and anchored on `nc` (north-center)

# %% tags=[]
text_metal3 = gf.partial(
    gf.components.text_rectangular_multi_layer, layers=(gf.LAYER.M3,)
)

m = gf.pack(sweep, text=text_metal3, text_anchors=("nc",), text_prefix="s")
c = m[0]
c

# %% tags=[]
text_metal2 = gf.partial(gf.components.text, layer=gf.LAYER.M2)

m = gf.pack(sweep, text=text_metal2, text_anchors=("nc",), text_prefix="s")
c = m[0]
c

# %% [markdown]
# ## Grid
#
# You can also pack components with a constant spacing.

# %% tags=[]
g = gf.grid(sweep)
g

# %% tags=[]
gh = gf.grid(sweep, shape=(1, len(sweep)))
gh

# %% tags=[]
gh_ymin = gf.grid(sweep, shape=(1, len(sweep)), align_y="ymin")
gh_ymin

# %% [markdown]
# You can also add text labels to each element of the sweep

# %% tags=[] vscode={"languageId": "markdown"}
gh_ymin = gf.grid_with_text(
    sweep, shape=(1, len(sweep)), align_y="ymin", text=text_metal3
)
gh_ymin

# %% [markdown]
# You can modify the text by customizing the `text_function` that you pass to `grid_with_text`

# %% tags=[] vscode={"languageId": "markdown"}
gh_ymin_m2 = gf.grid_with_text(
    sweep, shape=(1, len(sweep)), align_y="ymin", text=text_metal2
)
gh_ymin_m2

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

# %% tags=[] vscode={"languageId": "markdown"}
text_metal3 = gf.partial(
    gf.components.text_rectangular_multi_layer, layers=(gf.LAYER.M3,)
)
grid = gf.partial(gf.grid_with_text, text=text_metal3)
pack = gf.partial(gf.pack, text=text_metal3)

gratings_sweep = [
    gf.components.grating_coupler_elliptical(taper_angle=taper_angle)
    for taper_angle in [20, 30, 40]
]
gratings = grid(gratings_sweep, text=None)
gratings

# %% tags=[]
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
gratings

# %% tags=[]
sweep_resistance = [
    gf.components.resistance_sheet(width=width) for width in [1, 10, 100]
]
resistance = gf.pack(sweep_resistance)[0]
resistance

# %% tags=[]
spiral_te = gf.compose(
    gf.routing.add_fiber_single,
    gf.functions.rotate90,
    gf.components.spiral_inner_io_fiber_single,
)
sweep_spirals = [spiral_te(length=length) for length in [10e3, 20e3, 30e3]]
spirals = gf.pack(sweep_spirals)[0]
spirals

# %% tags=[]
mask = gf.pack([spirals, resistance, gratings])[0]
mask

# %% [markdown]
# As you can see you can define your mask in a single line.
#
# For more complex mask, you can also create a new cell to build up more complexity


# %% tags=[]
@gf.cell
def mask():
    c = gf.Component()
    c << gf.pack([spirals, resistance, gratings])[0]
    c << gf.components.seal_ring(c.bbox)
    return c


c = mask(cache=False)
c

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
# `pack_doe` places components as compact as possible

# %% [markdown]
# When running this tutorial make sure you UNCOMMENT this line `%matplotlib widget` so you can live update your changes in the YAML file
#
# `# %matplotlib widget`  -> `%matplotlib widget`

# %% tags=[]
# # %matplotlib widget

# %% tags=[]
x = ipywidgets.Textarea(rows=20, columns=480)

x.value = """
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

out = ipywidgets.Output()
display(x, out)


def f(change, out=out):
    try:
        c = gf.read.from_yaml(change["new"])
        # clear_output()
        fig = c.plot()
        c.show(show_ports=True)
        out.clear_output()
    except Exception as e:
        out.clear_output()
        with out:
            display(e)


x.observe(f, "value")
f({"new": x.value})

# %% [markdown]
# ### 2.2 pack_doe_grid
#
# `pack_doe_grid` places each component on a regular grid

# %% tags=[]
x.value = """
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

display(x, out)

# %% [markdown] tags=[]
# ## Metadata
#
# When saving GDS files is also convenient to store the metadata settings that you used to generate the GDS file.

# %% tags=[]
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
gdspath = c.write_gds_with_metadata("demo.gds")


# %% tags=[]
from IPython.display import Code

metadata = gdspath.with_suffix(".yml")
Code(metadata)

# %% tags=[]
import pandas as pd
from omegaconf import OmegaConf
import gdsfactory as gf


def mzi_te(**kwargs) -> gf.Component:
    gc = gf.c.grating_coupler_elliptical_tm()
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_array(c, grating_coupler=gc, **kwargs)
    c = gf.routing.add_electrical_pads_shortest(c)
    return c


c = mzi_te()
c

# %% tags=[]
c = gf.grid(
    [mzi_te()] * 2,
    decorator=gf.add_labels.add_labels_to_ports,
    add_ports_suffix=True,
    add_ports_prefix=False,
)
gdspath = c.write_gds()
csvpath = gf.labels.write_labels.write_labels_gdstk(gdspath, debug=True)

df = pd.read_csv(csvpath)
c

# %% tags=[]
df

# %% tags=[]
mzis = [mzi_te()] * 2
spirals = [
    gf.routing.add_fiber_array(gf.components.spiral_external_io(length=length))
    for length in [10e3, 20e3, 30e3]
]

c = gf.pack(
    mzis + spirals,
    add_ports_suffix=True,
    add_ports_prefix=False,
)[0]
c = gf.add_labels.add_labels_to_ports(c)
gdspath = c.write_gds()
csvpath = gf.labels.write_labels.write_labels_gdstk(gdspath, debug=True)
df = pd.read_csv(csvpath)
c

# %% tags=[]
gdspath = c.write_gds_with_metadata(gdsdir="extra")

# %% tags=[]
yaml_path = gdspath.with_suffix(".yml")

# %% tags=[]
labels_path = gf.labels.write_labels.write_labels_gdstk(
    gdspath=gdspath, layer_label=(201, 0)
)

# %% tags=[]
mask_metadata = OmegaConf.load(yaml_path)

# %% tags=[]
test_metadata = tm = gf.labels.merge_test_metadata(
    labels_path=labels_path, mask_metadata=mask_metadata
)

# %% tags=[]
tm.keys()

# %% [markdown]
# ```
#
# CSV labels  ------|
#                   |--> merge_test_metadata dict
#                   |
# YAML metadata  ---
#
# ```

# %% tags=[]
spiral_names = [s for s in test_metadata.keys() if s.startswith("spiral")]
spiral_names

# %% tags=[]
spiral_lengths = [
    test_metadata[spiral_name].info.length for spiral_name in spiral_names
]
spiral_lengths
