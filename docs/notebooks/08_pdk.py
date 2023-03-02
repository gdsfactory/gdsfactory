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
# # PDK
#
# gdsfactory includes a generic PDK, that you can use as an inspiration to create your own.
#
# What is a PDK? PDK stands for process design kit. It includes:
#
# 1. LayerStack: different layers with different thickness that come from a process.
# 2. DRC: Manufacturing rules.
# 3. A library of components. We use Parametric cells to generate components as well as fixed cells that always return the same component.
#
# The PDK allows you to register:
#
# - `cell` functions that return Components from a ComponentSpec (string, Component, ComponentFactory or dict). Also known as PCells (parametric cells).
# - `cross_section` functions that return CrossSection from a CrossSection Spec (string, CrossSection, CrossSectionFactory or dict).
# - `layers` that return a GDS Layer from a string, an int or a Tuple[int, int].
#
#
# You can only have one active PDK at a time.
# Thanks to PDK you can access components, cross_sections or layers using a string.
#
# Depending on the active pdk:
#
# - `get_layer` returns a Layer from the registered layers.
# - `get_component` returns a Component from the registered cells or containers.
# - `get_cross_section` returns a CrossSection from the registered cross_sections.

# %% [markdown]
# ## layers
#
# GDS layers are a tuple of two integer number `gdslayer/gdspurpose`
#
# You can define all the layers from your PDK:
#
# 1. From a Klayout `lyp` (layer properties file).
# 2. From scratch, adding all your layers into a class.
#
#
# Lets generate the layers definition code from a KLayout `lyp` file.

# %% tags=[]
import pathlib
from typing import Callable, Tuple

import pytest
from pydantic import BaseModel
from pytest_regressions.data_regression import DataRegressionFixture

from gdsfactory.add_pins import add_pin_rectangle_inside
from gdsfactory.component import Component
from gdsfactory.config import PATH
from gdsfactory.cross_section import cross_section
from gdsfactory.decorators import flatten_invalid_refs, has_valid_transformations
from gdsfactory.difftest import difftest
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.technology import (
    LayerLevel,
    LayerStack,
    LayerView,
    LayerViews,
    lyp_to_dataclass,
)
from gdsfactory.typings import Layer, LayerSpec

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

print(lyp_to_dataclass(PATH.klayout_lyp))


# %% tags=[]
class LayerMap(BaseModel):
    WG: Layer = (1, 0)
    DEVREC: Layer = (68, 0)
    PORT: Layer = (1, 10)
    PORTE: Layer = (1, 11)
    LABEL: Layer = (201, 0)
    LABEL_INSTANCES: Layer = (206, 0)
    LABEL_SETTINGS: Layer = (202, 0)
    LUMERICAL: Layer = (733, 0)
    M1: Layer = (41, 0)
    M2: Layer = (45, 0)
    M3: Layer = (49, 0)
    N: Layer = (20, 0)
    NP: Layer = (22, 0)
    NPP: Layer = (24, 0)
    OXIDE_ETCH: Layer = (6, 0)
    P: Layer = (21, 0)
    PDPP: Layer = (27, 0)
    PP: Layer = (23, 0)
    PPP: Layer = (25, 0)
    PinRec: Layer = (1, 10)
    PinRecM: Layer = (1, 11)
    SHALLOWETCH: Layer = (2, 6)
    SILICIDE: Layer = (39, 0)
    SIM_REGION: Layer = (100, 0)
    SITILES: Layer = (190, 0)
    SLAB150: Layer = (2, 0)
    SLAB150CLAD: Layer = (2, 9)
    SLAB90: Layer = (3, 0)
    SLAB90CLAD: Layer = (3, 1)
    SOURCE: Layer = (110, 0)
    TE: Layer = (203, 0)
    TEXT: Layer = (66, 0)
    TM: Layer = (204, 0)
    Text: Layer = (66, 0)
    VIA1: Layer = (44, 0)
    VIA2: Layer = (43, 0)
    VIAC: Layer = (40, 0)
    WGCLAD: Layer = (111, 0)
    WGN: Layer = (34, 0)
    WGNCLAD: Layer = (36, 0)

    class Config:
        frozen = True
        extra = "forbid"


LAYER = LayerMap()

# %% [markdown]
# There are some default layers in some generic components and cross_sections, that it may be convenient adding.
#
# | Layer          | Purpose                                                      |
# | -------------- | ------------------------------------------------------------ |
# | DEVREC         | device recognition layer. For connectivity checks.           |
# | PORT           | optical port pins. For connectivity checks.                  |
# | PORTE          | electrical port pins. For connectivity checks.               |
# | SHOW_PORTS     | add port pin markers when `Component.show(show_ports=True)`  |
# | LABEL          | for adding labels to grating couplers for automatic testing. |
# | LABEL_INSTANCE | for adding instance labels on `gf.read.from_yaml`            |
# | TE             | for TE polarization fiber marker.                            |
# | TM             | for TM polarization fiber marker.                            |
#
#
# ```python
# class LayersConvenient(BaseModel):
#     DEVREC: Layer = (68, 0)
#     PORT: Layer = (1, 10)  # PinRec optical
#     PORTE: Layer = (1, 11)  # PinRec electrical
#     SHOW_PORTS: Layer = (1, 12)
#
#     LABEL: Layer = (201, 0)
#     LABEL_INSTANCE: Layer = (66, 0)
#     TE: Layer = (203, 0)
#     TM: Layer = (204, 0)
#
# ```

# %% [markdown]
# ## cross_sections
#
# You can create a `CrossSection` from scratch or you can customize the cross_section functions in `gf.cross_section`

# %% tags=[]
strip2 = gf.partial(gf.cross_section.strip, layer=(2, 0))

# %% tags=[]
c = gf.components.straight(cross_section=strip2)
c

# %% tags=[]
pin = gf.partial(
    gf.cross_section.strip,
    sections=(
        gf.Section(width=2, layer=gf.LAYER.N, offset=+1),
        gf.Section(width=2, layer=gf.LAYER.P, offset=-1),
    ),
)
c = gf.components.straight(cross_section=pin)
c

# %% tags=[]
strip_wide = gf.partial(gf.cross_section.strip, width=3)


# %% tags=[]
strip = gf.partial(
    gf.cross_section.strip, auto_widen=True
)  # auto_widen tapers to wider waveguides for lower loss in long straight sections.

# %% tags=[]
cross_sections = dict(strip_wide=strip_wide, pin=pin, strip=strip)

# %% [markdown]
# ## cells
#
# Cells are functions that return Components. They are parametrized and accept also cells as parameters, so you can build many levels of complexity. Cells are also known as PCells or parametric cells.
#
# You can customize the function default arguments easily thanks to `functools.partial`
# Lets customize the default arguments of a library of cells.
#
# For example, you can make some wide MMIs for a particular technology. Lets say the best MMI width you found it to be 9um.

# %% tags=[]
mmi1x2 = gf.partial(gf.components.mmi1x2, width_mmi=9)
mmi2x2 = gf.partial(gf.components.mmi2x2, width_mmi=9)

cells = dict(mmi1x2=mmi1x2, mmi2x2=mmi2x2)

# %% [markdown]
# ## PDK
#
# You can register Layers, ComponentFactories (Parametric cells) and CrossSectionFactories (cross_sections) into a PDK. Then you can access them by a string after you activate the pdk `PDK.activate()`.
#
# ### LayerSpec
#
# You can access layers from the active Pdk using the layer name or a tuple/list of two numbers.

# %% tags=[]
from gdsfactory.generic_tech import get_generic_pdk

generic_pdk = get_generic_pdk()

pdk1 = gf.Pdk(
    name="fab1",
    layers=LAYER.dict(),
    cross_sections=cross_sections,
    cells=cells,
    base_pdk=generic_pdk,
    sparameters_path=gf.config.sparameters_path,
    layer_views=generic_pdk.layer_views,
)
pdk1.activate()

# %% tags=[]
pdk1.get_layer("WG")

# %% tags=[]
pdk1.get_layer([1, 0])

# %% [markdown]
# ### CrossSectionSpec
#
# You can access cross_sections from the pdk from the cross_section name, or using a dict to customize the CrossSection

# %% tags=[]
pdk1.get_cross_section("pin")

# %% tags=[]
cross_section_spec_string = "pin"
gf.components.straight(cross_section=cross_section_spec_string)

# %% tags=[]
cross_section_spec_dict = dict(cross_section="pin", settings=dict(width=2))
print(pdk1.get_cross_section(cross_section_spec_dict))
wg_pin = gf.components.straight(cross_section=cross_section_spec_dict)
wg_pin

# %% [markdown]
# ### ComponentSpec
#
# You can get Component from the active pdk using the cell name (string) or a dict.

# %% tags=[]
pdk1.get_component("mmi1x2")

# %% tags=[]
pdk1.get_component(dict(component="mmi1x2", settings=dict(length_mmi=10)))

# %% [markdown]
# Now you can define PDKs for different Fabs
#
# ### FabA
#
# FabA only has one waveguide layer available that is defined in GDS layer (30, 0)
#
# The waveguide traces are 2um wide.

# %% tags=[]
nm = 1e-3


class LayerMap(BaseModel):
    WG: Layer = (34, 0)
    SLAB150: Layer = (2, 0)
    DEVREC: Layer = (68, 0)
    PORT: Layer = (1, 10)
    PORTE: Layer = (1, 11)
    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    TEXT: Layer = (66, 0)


LAYER = LayerMap()


class FabALayerViews(LayerViews):
    WG = LayerView(color="gold")
    SLAB150 = LayerView(color="red")
    TE = LayerView(color="green")


LAYER_VIEWS = FabALayerViews(layer_map=LAYER.dict())


def get_layer_stack_faba(
    thickness_wg: float = 500 * nm, thickness_slab: float = 150 * nm
) -> LayerStack:
    """Returns fabA LayerStack"""

    class FabALayerStack(LayerStack):
        strip = LayerLevel(
            layer=LAYER.WG,
            thickness=thickness_wg,
            zmin=0.0,
            material="si",
        )
        strip2 = LayerLevel(
            layer=LAYER.SLAB150,
            thickness=thickness_slab,
            zmin=0.0,
            material="si",
        )

    return FabALayerStack()


LAYER_STACK = get_layer_stack_faba()

WIDTH = 2

# Specify a cross_section to use
strip = gf.partial(gf.cross_section.cross_section, width=WIDTH, layer=LAYER.WG)

mmi1x2 = gf.partial(
    gf.components.mmi1x2,
    width=WIDTH,
    width_taper=WIDTH,
    width_mmi=3 * WIDTH,
    cross_section=strip,
)

generic_pdk = gf.generic_tech.get_generic_pdk()

fab_a = gf.Pdk(
    name="Fab_A",
    cells=dict(mmi1x2=mmi1x2),
    cross_sections=dict(strip=strip),
    layers=LAYER.dict(),
    base_pdk=generic_pdk,
    sparameters_path=gf.config.sparameters_path,
    layer_views=LAYER_VIEWS,
    layer_stack=LAYER_STACK,
)
fab_a.activate()

gc = gf.partial(
    gf.components.grating_coupler_elliptical_te, layer=LAYER.WG, cross_section=strip
)

c = gf.components.mzi()
c_gc = gf.routing.add_fiber_array(component=c, grating_coupler=gc, with_loopback=False)
c_gc.plot()

# %% tags=[]
c = c_gc.to_3d()
c.show(show_ports=True)

# %% [markdown]
# ### FabB
#
# FabB has photonic waveguides that require rectangular cladding layers to avoid dopants
#
# Lets say that the waveguides are defined in layer (2, 0) and are 0.3um wide, 1um thick
#

# %% tags=[]
nm = 1e-3


class LayerMap(BaseModel):
    WG: Layer = (2, 0)
    SLAB150: Layer = (3, 0)
    DEVREC: Layer = (68, 0)
    PORT: Layer = (1, 10)
    PORTE: Layer = (1, 11)
    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    TEXT: Layer = (66, 0)
    LABEL: Layer = (201, 0)
    DOPING_BLOCK1: Layer = (61, 0)
    DOPING_BLOCK2: Layer = (62, 0)


LAYER = LayerMap()


# The LayerViews class supports grouping LayerViews within each other. These groups are maintained when exporting a LayerViews object to a KLayout layer properties (.lyp) file.
class FabBLayerViews(LayerViews):
    WG = LayerView(color="red")
    SLAB150 = LayerView(color="blue")
    TE = LayerView(color="green")
    PORT = LayerView(color="green", alpha=0)

    class DopingBlockGroup(LayerView):
        DOPING_BLOCK1 = LayerView(color="green", alpha=0)
        DOPING_BLOCK2 = LayerView(color="green", alpha=0)

    DopingBlocks = DopingBlockGroup()


LAYER_VIEWS = FabBLayerViews(layer_map=LAYER)


def get_layer_stack_fab_b(
    thickness_wg: float = 1000 * nm, thickness_slab: float = 150 * nm
) -> LayerStack:
    """Returns fabA LayerStack."""

    class FabBLayerStack(LayerStack):
        strip = LayerLevel(
            layer=LAYER.WG,
            thickness=thickness_wg,
            zmin=0.0,
            material="si",
        )
        strip2 = LayerLevel(
            layer=LAYER.SLAB150,
            thickness=thickness_slab,
            zmin=0.0,
            material="si",
        )

    return FabBLayerStack()


LAYER_STACK = get_layer_stack_fab_b()


WIDTH = 0.3
BBOX_LAYERS = (LAYER.DOPING_BLOCK1, LAYER.DOPING_BLOCK2)
BBOX_OFFSETS = (3, 3)

# use cladding_layers and cladding_offsets if the foundry prefers conformal blocking doping layers instead of squared
# bbox_layers and bbox_offsets makes rectangular waveguides.
strip = gf.partial(
    gf.cross_section.cross_section,
    width=WIDTH,
    layer=LAYER.WG,
    # bbox_layers=BBOX_LAYERS,
    # bbox_offsets=BBOX_OFFSETS,
    cladding_layers=BBOX_LAYERS,
    cladding_offsets=BBOX_OFFSETS,
)

straight = gf.partial(gf.components.straight, cross_section=strip)
bend_euler = gf.partial(gf.components.bend_euler, cross_section=strip)
mmi1x2 = gf.partial(
    gf.components.mmi1x2,
    cross_section=strip,
    width=WIDTH,
    width_taper=WIDTH,
    width_mmi=4 * WIDTH,
)
mzi = gf.partial(gf.components.mzi, cross_section=strip, splitter=mmi1x2)
gc = gf.partial(
    gf.components.grating_coupler_elliptical_te, layer=LAYER.WG, cross_section=strip
)

cells = dict(
    gc=gc,
    mzi=mzi,
    mmi1x2=mmi1x2,
    bend_euler=bend_euler,
    straight=straight,
    taper=gf.components.taper,
)
cross_sections = dict(strip=strip)

pdk = gf.Pdk(
    name="fab_b",
    cells=cells,
    cross_sections=cross_sections,
    layers=LAYER.dict(),
    sparameters_path=gf.config.sparameters_path,
    layer_views=LAYER_VIEWS,
    layer_stack=LAYER_STACK,
)
pdk.activate()


c = mzi()
wg_gc = gf.routing.add_fiber_array(
    component=c, grating_coupler=gc, cross_section=strip, with_loopback=False
)
wg_gc.plot()

# %% tags=[]
c = wg_gc.to_3d()
c.show(show_ports=True)

# %% [markdown]
# ### FabC
#
# Lets assume that fab C has similar technology to the generic PDK in gdsfactory and that you just want to remap some layers, and adjust the widths.
#

# %% tags=[]
nm = 1e-3


class LayerMap(BaseModel):
    WG: Layer = (10, 1)
    WG_CLAD: Layer = (10, 2)
    WGN: Layer = (34, 0)
    WGN_CLAD: Layer = (36, 0)
    SLAB150: Layer = (2, 0)
    DEVREC: Layer = (68, 0)
    PORT: Layer = (1, 10)
    PORTE: Layer = (1, 11)
    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    TEXT: Layer = (66, 0)
    LABEL: Layer = (201, 0)


LAYER = LayerMap()
WIDTH_NITRIDE_OBAND = 0.9
WIDTH_NITRIDE_CBAND = 1.0
PORT_TYPE_TO_LAYER = dict(optical=(100, 0))


# This is something you usually define in KLayout
class FabCLayerViews(LayerViews):
    WG = LayerView(color="black")
    SLAB150 = LayerView(color="blue")
    WGN = LayerView(color="orange")
    WGN_CLAD = LayerView(color="blue", alpha=0, visible=False)

    class SimulationGroup(LayerView):
        TE = LayerView(color="green")
        PORT = LayerView(color="green", alpha=0)

    Simulation = SimulationGroup()

    class DopingGroup(LayerView):
        DOPING_BLOCK1 = LayerView(color="green", alpha=0, visible=False)
        DOPING_BLOCK2 = LayerView(color="green", alpha=0, visible=False)

    Doping = DopingGroup()


LAYER_VIEWS = FabCLayerViews(layer_map=LAYER)


def get_layer_stack_fab_c(
    thickness_wg: float = 350.0 * nm, thickness_clad: float = 3.0
) -> LayerStack:
    """Returns generic LayerStack"""

    class FabCLayerStack(LayerStack):
        core = LayerLevel(
            layer=LAYER.WGN,
            thickness=thickness_wg,
            zmin=0,
        )
        clad = LayerLevel(
            layer=LAYER.WGN_CLAD,
            thickness=thickness_clad,
            zmin=0,
        )

    return FabCLayerStack()


LAYER_STACK = get_layer_stack_fab_c()


def add_pins(
    component: Component,
    function: Callable = add_pin_rectangle_inside,
    pin_length: float = 0.5,
    port_layer: Layer = LAYER.PORT,
    **kwargs,
) -> Component:
    """Add Pin port markers.

    Args:
        component: to add ports.
        function: to add pins.
        pin_length: in um.
        port_layer: spec.
        kwargs: function kwargs.
    """
    for p in component.ports.values():
        function(
            component=component,
            port=p,
            layer=port_layer,
            layer_label=port_layer,
            pin_length=pin_length,
            **kwargs,
        )
    return component


# cross_section constants
bbox_layers = [LAYER.WGN_CLAD]
bbox_offsets = [3]

# Nitride Cband
xs_nc = gf.partial(
    cross_section,
    width=WIDTH_NITRIDE_CBAND,
    layer=LAYER.WGN,
    bbox_layers=bbox_layers,
    bbox_offsets=bbox_offsets,
    add_pins=add_pins,
)
# Nitride Oband
xs_no = gf.partial(
    cross_section,
    width=WIDTH_NITRIDE_OBAND,
    layer=LAYER.WGN,
    bbox_layers=bbox_layers,
    bbox_offsets=bbox_offsets,
    add_pins=add_pins,
)


cross_sections = dict(xs_nc=xs_nc, xs_no=xs_no, strip=xs_nc)

# LEAF cells have pins
mmi1x2_nc = gf.partial(
    gf.components.mmi1x2,
    width=WIDTH_NITRIDE_CBAND,
    cross_section=xs_nc,
)
mmi1x2_no = gf.partial(
    gf.components.mmi1x2,
    width=WIDTH_NITRIDE_OBAND,
    cross_section=xs_no,
)
bend_euler_nc = gf.partial(
    gf.components.bend_euler,
    cross_section=xs_nc,
)
straight_nc = gf.partial(
    gf.components.straight,
    cross_section=xs_nc,
)
bend_euler_no = gf.partial(
    gf.components.bend_euler,
    cross_section=xs_no,
)
straight_no = gf.partial(
    gf.components.straight,
    cross_section=xs_no,
)

gc_nc = gf.partial(
    gf.components.grating_coupler_elliptical_te,
    grating_line_width=0.6,
    layer=LAYER.WGN,
    cross_section=xs_nc,
)

# HIERARCHICAL cells are made of leaf cells
mzi_nc = gf.partial(
    gf.components.mzi,
    cross_section=xs_nc,
    splitter=mmi1x2_nc,
    straight=straight_nc,
    bend=bend_euler_nc,
)
mzi_no = gf.partial(
    gf.components.mzi,
    cross_section=xs_no,
    splitter=mmi1x2_no,
    straight=straight_no,
    bend=bend_euler_no,
)


cells = dict(
    mmi1x2_nc=mmi1x2_nc,
    mmi1x2_no=mmi1x2_no,
    bend_euler_nc=bend_euler_nc,
    bend_euler_no=bend_euler_no,
    straight_nc=straight_nc,
    straight_no=straight_no,
    gc_nc=gc_nc,
    mzi_nc=mzi_nc,
    mzi_no=mzi_no,
)

pdk = gf.Pdk(
    name="fab_c",
    cells=cells,
    cross_sections=cross_sections,
    layers=LAYER.dict(),
    sparameters_path=gf.config.sparameters_path,
    layer_views=LAYER_VIEWS,
    layer_stack=LAYER_STACK,
)
pdk.activate()


# %% tags=[]
LAYER_VIEWS.layer_map.values()

# %% tags=[]
mzi = mzi_nc()
mzi_gc = gf.routing.add_fiber_single(
    component=mzi,
    grating_coupler=gc_nc,
    cross_section=xs_nc,
    optical_routing_type=1,
    straight=straight_nc,
    bend=bend_euler_nc,
)
mzi_gc.plot()

# %% tags=[]
c = mzi_gc.to_3d()
c.show(show_ports=True)

# %% tags=[]
ls = get_layer_stack_fab_c()

# %% [markdown]
# ## Testing PDK cells
#
# To make sure all your PDK PCells produce the components that you want, it's important to test your PDK cells.
#
# As you write your own cell functions you want to make sure you do not break or produced unwanted regressions later on. You should write tests for this.
#
# Make sure you create a `test_components.py` file for pytest to test your PDK. See for example the tests in the [ubc PDK](https://github.com/gdsfactory/ubc)
#
# Pytest-regressions automatically creates the CSV and YAML files for you, as well `gdsfactory.gdsdiff` will store the reference GDS in ref_layouts and check for geometry differences using XOR.
#
# gdsfactory is **not** backwards compatible, which means that the package will keep improving and evolving.
#
# 1. To make your work stable you should install a specific version and [pin the version](https://martin-thoma.com/python-requirements/) in your `requirements.txt` or `pyproject.toml` as `gdsfactory==6.48.3` replacing `6.48.3` by whatever version you end up using.
# 2. Before you upgrade gdsfactory to a newer version make sure your tests pass to make sure that things behave as expected
#
#

# %% tags=[]
"""This code tests all your cells in the PDK

it will test 3 things:

1. difftest: will test the GDS geometry of a new GDS compared to a reference. Thanks to Klayout fast booleans.add()
2. settings test: will compare the settings in YAML with a reference YAML file.add()
3. ensure ports are on grid, to avoid port snapping errors that can create 1nm gaps later on when you build circuits.

"""

try:
    dirpath = pathlib.Path(__file__).absolute().with_suffix(".gds")
except Exception:
    dirpath = pathlib.Path.cwd()


component_names = list(pdk.cells.keys())
factory = pdk.cells


@pytest.fixture(params=component_names, scope="function")
def component_name(request) -> str:
    return request.param


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS names, shapes and layers.
    Runs XOR and computes the area."""
    component = factory[component_name]()
    test_name = f"fabc_{component_name}"
    difftest(component, test_name=test_name, dirpath=dirpath)


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions in component settings and ports."""
    component = factory[component_name]()
    data_regression.check(component.to_dict())


def test_assert_ports_on_grid(component_name: str):
    """Ensures all ports are on grid to avoid 1nm gaps"""
    component = factory[component_name]()
    component.assert_ports_on_grid()


# %% [markdown]
# ## Compare gds files
#
# You can use the command line `gf gds diff gds1.gds gds2.gds` to overlay `gds1.gds` and `gds2.gds` files and show them in KLayout.
#
# For example, if you changed the mmi1x2 and made it 5um longer by mistake, you could `gf gds diff ref_layouts/mmi1x2.gds run_layouts/mmi1x2.gds` and see the GDS differences in Klayout.

# %% [markdown]
# ## PDK decorator
#
# You can also define a PDK decorator (function) that runs over every PDK PCell.

# %%
from gdsfactory.add_pins import add_pins_siepic


def add_pins_bbox_siepic(
    component: Component,
    port_type: str = "optical",
    layer_pin: LayerSpec = "PORT",
    pin_length: float = 2 * nm,
    bbox_layer: LayerSpec = "DEVREC",
    padding: float = 0,
) -> Component:
    """Add bounding box device recognition layer and pins.

    Args:
        component: to add pins.
        function: to add pins.
        port_type: optical, electrical...
        layer_pin: for pin.
        pin_length: in um.
        bbox_layer: bounding box layer.
        padding: around device.
    """
    c = component
    c.add_padding(default=padding, layers=(bbox_layer,))
    c = add_pins_siepic(
        component=component,
        port_type=port_type,
        layer_pin=layer_pin,
        pin_length=pin_length,
    )
    return c


pdk = gf.Pdk(
    name="fab_c",
    cells=cells,
    cross_sections=cross_sections,
    layers=LAYER.dict(),
    sparameters_path=gf.config.sparameters_path,
    layer_views=LAYER_VIEWS,
    layer_stack=LAYER_STACK,
    # default_decorator=add_pins_bbox_siepic
)
pdk.activate()

c1 = gf.components.straight(length=5)
print(has_valid_transformations(c1))
c1.layers

# %%
pdk = gf.Pdk(
    name="fab_c",
    cells=cells,
    cross_sections=cross_sections,
    layers=LAYER.dict(),
    sparameters_path=gf.config.sparameters_path,
    layer_views=LAYER_VIEWS,
    layer_stack=LAYER_STACK,
    default_decorator=add_pins_bbox_siepic,
)
pdk.activate()

c1 = gf.components.straight(length=5)
print(has_valid_transformations(c1))
c1.layers

# %% [markdown]
# if you zoom in you will see a device recognition layer and pins.
#
# ![devrec](https://i.imgur.com/U9IPOei.png)
#

# %% [markdown]
# ## Version control components
#
# For version control your component library you can use GIT
#
# For tracking changes you can add `Component` changelog in the PCell docstring.

# %%
from gdsfactory.generic_tech import get_generic_pdk

PDK = get_generic_pdk()
PDK.activate()


@gf.cell
def litho_ruler(
    height: float = 2,
    width: float = 0.5,
    spacing: float = 2.0,
    scale: Tuple[float, ...] = (3, 1, 1, 1, 1, 2, 1, 1, 1, 1),
    num_marks: int = 21,
    layer: LayerSpec = (1, 0),
) -> gf.Component:
    """Ruler structure for lithographic measurement.

    Includes marks of varying scales to allow for easy reading by eye.

    Args:
        height: Height of the ruling marks in um.
        width: Width of the ruling marks in um.
        spacing: Center-to-center spacing of the ruling marks in um.
        scale: Height scale pattern of marks.
        num_marks: Total number of marks to generate.
        layer: Specific layer to put the ruler geometry on.
    """
    D = gf.Component()
    for n in range(num_marks):
        h = height * scale[n % len(scale)]
        D << gf.components.rectangle(size=(width, h), layer=layer)

    D.distribute(direction="x", spacing=spacing, separation=False, edge="x")
    D.align(alignment="ymin")
    return D


c = litho_ruler(cache=False)
c.plot()

# %% [markdown]
# Lets assume that later on you change the code inside the PCell and want to keep a changelog.
# You can use the docstring Notes to document any significant changes in the component.


# %%
@gf.cell
def litho_ruler(
    height: float = 2,
    width: float = 0.5,
    spacing: float = 2.0,
    scale: Tuple[float, ...] = (3, 1, 1, 1, 1, 2, 1, 1, 1, 1),
    num_marks: int = 21,
    layer: LayerSpec = (1, 0),
) -> gf.Component:
    """Ruler structure for lithographic measurement.

    Args:
        height: Height of the ruling marks in um.
        width: Width of the ruling marks in um.
        spacing: Center-to-center spacing of the ruling marks in um.
        scale: Height scale pattern of marks.
        num_marks: Total number of marks to generate.
        layer: Specific layer to put the ruler geometry on.

    Notes:
        5.6.7: distribute across y instead of x.
    """
    D = gf.Component()
    for n in range(num_marks):
        h = height * scale[n % len(scale)]
        ref = D << gf.components.rectangle(size=(width, h), layer=layer)
        ref.rotate(90)

    D.distribute(direction="y", spacing=spacing, separation=False, edge="y")
    D.align(alignment="xmin")
    return D


c = litho_ruler(cache=False)
c.plot()
