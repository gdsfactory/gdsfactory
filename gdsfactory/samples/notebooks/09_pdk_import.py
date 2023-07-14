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
# # Import PDK
#
# ## Import PDK from GDS files
#
# To import a PDK from GDS files into gdsfactory you need:
#
# - GDS file with all the cells that you want to import in the PDK (or separate GDS files, where each file contains a GDS design).
#
# Ideally you also get:
#
# - Klayout layer properties files, to define the Layers that you can use when creating new custom Components. This allows you to define the LayerMap that maps Layer_name to (GDS_LAYER, GDS_PuRPOSE).
# - layer_stack information (material index, thickness, z positions of each layer).
# - DRC rules. If you don't get this you can easily build one using klayout.
#
# GDS files are great for describing geometry thanks to the concept of References, where you store any geometry only once in memory.
#
# For storing device metadata (settings, port locations, port widths, port angles ...) there is no clear standard.
#
# `gdsfactory` stores the that metadata in `YAML` files, and also has functions to add pins
#
# - `Component.write_gds()` saves GDS
# - `Component.write_gds_metadata()` save GDS + YAML metadata

# %%

# Lets generate the script that we need to have to each GDS cell into gdsfactory
from gdsfactory.config import PATH
from gdsfactory.technology import lyp_to_dataclass
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

c = gf.components.mzi()
c.plot()

# %% [markdown]
# You can write **GDS** files only

# %%
gdspath = c.write_gds("extra/mzi.gds")

# %% [markdown]
# Or **GDS** with **YAML** metadata information (ports, settings, cells ...)

# %%
gdspath = c.write_gds("extra/mzi.gds", with_metadata=True)

# %% [markdown]
# This created a `mzi.yml` file that contains:
# - ports
# - cells (flat list of cells)
# - info (function name, module, changed settings, full settings, default settings)

# %%
c.metadata.keys()

# %% [markdown]
# You can read GDS files into gdsfactory thanks to the `import_gds` function
#
# `import_gds` reads the same GDS file from disk without losing any information

# %%
gf.clear_cache()

c = gf.import_gds(gdspath, read_metadata=True)
c.plot()

# %%
c2 = gf.import_gds(gdspath, name="mzi_sample", read_metadata=True)
c2.plot()

# %%
c2.name

# %%
c3 = gf.routing.add_fiber_single(c2)
c3.plot()

# %%
gdspath = c3.write_gds("extra/pdk.gds", with_metadata=True)

# %%
gf.labels.write_labels.write_labels_klayout(gdspath, layer_label=gf.LAYER.LABEL)

# %% [markdown]
# ### add ports from pins
#
# Sometimes the GDS does not have YAML metadata, therefore you need to figure out the port locations, widths and orientations.
#
# gdsfactory provides you with functions that will add ports to the component by looking for pins shapes on a specific layers (port_markers or pins)
#
# There are different pin standards supported to automatically add ports to components:
#
# - PINs towards the inside of the port (port at the outer part of the PIN)
# - PINs with half of the pin inside and half outside (port at the center of the PIN)
# - PIN with only labels (no shapes). You have to manually specify the width of the port.
#
#
# Lets add pins, save a GDS and then import it back.

# %%
c = gf.components.straight(
    decorator=gf.add_pins.add_pins
)  # add pins inside the component
c.plot()

# %%
gdspath = c.write_gds("extra/wg.gds")

# %%
gf.clear_cache()
c2 = gf.import_gds(gdspath)
c2

# %%
c2.ports  # import_gds does not automatically add the pins

# %%
c3 = gf.import_gds(gdspath, decorator=gf.add_ports.add_ports_from_markers_inside)
c3

# %%
c3.ports

# %% [markdown]
# Foundries provide PDKs in different formats and commercial tools.
#
# The easiest way to import a PDK into gdsfactory is to:
#
# 1. have each GDS cell into a separate GDS file
# 2. have one GDS file with all the cells inside
# 3. Have a KLayout layermap. Makes easier to create the layermap.
#
# With that you can easily create the PDK as as python package.
#
# Thanks to having a gdsfactory PDK as a python package you can:
#
# - version control your PDK using GIT to keep track of changes and work on a team
#     - write tests of your pdk components to avoid unwanted changes from one component to another.
#     - ensure you maintain the quality of the PDK with continuous integration checks
#     - pin the version of gdsfactory, so new updates of gdsfactory won't affect your code
# - name your PDK version using [semantic versioning](https://semver.org/). For example patches increase the last number (0.0.1 -> 0.0.2)
# - install your PDK easily `pip install pdk_fab_a` and easily interface with other tools
#
#
#
# To create a **Python** package you can start from a customizable template (thanks to cookiecutter)
#
# You can create a python package by running this 2 commands inside a terminal:
#
# ```
# pip install cookiecutter
# cookiecutter https://github.com/joamatab/cookiecutter-pypackage-minimal
# ```
#
# It will ask you some questions to fill in the template (name of the package being the most important)
#
#
# Then you can add the information about the GDS files and the Layers inside that package

# %%
print(lyp_to_dataclass(PATH.klayout_lyp))

# %%
# lets create a sample PDK (for demo purposes only) using GDSfactory
# if the PDK is in a commercial tool you can also do this. Make sure you save a single pdk.gds

sample_pdk_cells = gf.grid(
    [
        gf.components.straight,
        gf.components.bend_euler,
        gf.components.grating_coupler_elliptical,
    ]
)
sample_pdk_cells.write_gds("extra/pdk.gds")
sample_pdk_cells

# %%
sample_pdk_cells.get_dependencies()

# %%
# we write the sample PDK into a single GDS file
gf.clear_cache()
gf.write_cells.write_cells(
    gdspath="extra/pdk.gds", dirpath="extra/gds", recursively=True
)

# %%
print(gf.write_cells.get_import_gds_script("extra/gds"))

# %% [markdown]
# You can also include the code to plot each fix cell in the docstring.

# %%
print(gf.write_cells.get_import_gds_script("extra/gds", module="samplepdk.components"))

# %% [markdown]
# ## Import PDK from other python packages
#
# You can Write the cells to GDS and use the
#
# Ideally you also start transitioning your legacy code Pcells into gdsfactory syntax. It's a great way to learn the gdsfactory way!
#
# Here is some advice:
#
# - Ask your foundry for the gdsfactory PDK.
# - Leverage the generic pdk cells available in gdsfactory.
# - Write tests for your cells.
# - Break the cells into small reusable functions.
# - use GIT to track changes.
# - review your code with your colleagues and other gdsfactory developers to get feedback. This is key to get better at coding gdsfactory.
# - get rid of any warnings you see.

# %% [markdown]
# ## Import PDK from YAML uPDK
#
# gdsfactory supports read and write to [uPDK YAML definition](https://openepda.org/index.html)
#
# Lets write a PDK into uPDK YAML definition and then convert it back to a gdsfactory script.
#
# the uPDK extracts the code from the docstrings.
#
# ```python
#
# def evanescent_coupler_sample() -> None:
#     """Evanescent coupler example.
#
#     Args:
#       coupler_length: length of coupling (min: 0.0, max: 200.0, um).
#     """
#     pass
#
# ```

# %%
from gdsfactory.samples.pdk.fab_c import pdk

yaml_pdk_decription = pdk.to_updk()
print(yaml_pdk_decription)


# %%
from gdsfactory.read.from_updk import from_updk

gdsfactory_script = from_updk(yaml_pdk_decription)
print(gdsfactory_script)

# %% [markdown]
# ## Build your own PDK
#
# You can create a PDK as a python library using a cookiecutter template. For example, you can use this one.
#
# ```
# pip install cookiecutter
# cookiecutter https://github.com/joamatab/cookiecutter-pypackage-minimal
# ```
#
# Or you can fork the ubcpdk and create new PCell functions that use the correct layers for your foundry. For example.
#
# ```
#
# from pydantic import BaseModel
#
#
# class LayerMap(BaseModel):
#     WGCORE = (3, 0)
#     LABEL = (100, 0)
#     DEVREC: Layer = (68, 0)
#     LABEL: Layer = (10, 0)
#     PORT: Layer = (1, 10)  # PinRec
#     PORTE: Layer = (1, 11)  # PinRecM
#     FLOORPLAN: Layer = (99, 0)
#
#     TE: Layer = (203, 0)
#     TM: Layer = (204, 0)
#     TEXT: Layer = (66, 0)
#     LABEL_INSTANCE: Layer = (66, 0)
#
#
# LAYER = LayerMap()
#
# ```
