# # PDK examples
#
# Different PDKs have different component libraries, design rules and layer stacks (GDS layers, materials and thickness).
#
# When you install a PDK you have to make sure you also installed the correct version of gdsfactory.
#
# Notice that some PDKs may have require different gdsfactory versions.

# +
from functools import partial
from typing import Callable

from pydantic import BaseModel

from gdsfactory.add_pins import add_pin_rectangle_inside
from gdsfactory.component import Component
from gdsfactory.config import CONF
from gdsfactory.cross_section import cross_section
from gdsfactory.technology import (
    LayerLevel,
    LayerStack,
    LayerView,
    LayerViews,
)
from gdsfactory.typings import Layer
import gdsfactory as gf

gf.config.rich_output()
nm = 1e-3


# +
CONF.display_type = "klayout"

p = gf.get_active_pdk()
p.name
# -

gf.config.print_version()

gf.config.print_version_pdks()
#

# ### FabA
#
# FabA only has one waveguide layer available that is defined in GDS layer (30, 0)
#
# The waveguide traces are 2um wide.


# +
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
strip = partial(gf.cross_section.cross_section, width=WIDTH, layer=LAYER.WG)

mmi1x2 = partial(
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

gc = partial(
    gf.components.grating_coupler_elliptical_te, layer=LAYER.WG, cross_section=strip
)

c = gf.components.mzi()
c_gc = gf.routing.add_fiber_array(component=c, grating_coupler=gc, with_loopback=False)
c_gc
# -

scene = c_gc.to_3d()
scene.show(show_ports=True)

# ### FabB
#
# FabB has photonic waveguides that require rectangular cladding layers to avoid dopants
#
# Lets say that the waveguides are defined in layer (2, 0) and are 0.3um wide, 1um thick
#

# +
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


# The LayerViews class supports grouping LayerViews within each other.
# These groups are maintained when exporting a LayerViews object to a KLayout layer properties (.lyp) file.
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
strip = partial(
    gf.cross_section.cross_section,
    width=WIDTH,
    layer=LAYER.WG,
    # bbox_layers=BBOX_LAYERS,
    # bbox_offsets=BBOX_OFFSETS,
    cladding_layers=BBOX_LAYERS,
    cladding_offsets=BBOX_OFFSETS,
)

straight = partial(gf.components.straight, cross_section=strip)
bend_euler = partial(gf.components.bend_euler, cross_section=strip)
mmi1x2 = partial(
    gf.components.mmi1x2,
    cross_section=strip,
    width=WIDTH,
    width_taper=WIDTH,
    width_mmi=4 * WIDTH,
)
mzi = partial(gf.components.mzi, cross_section=strip, splitter=mmi1x2)
gc = partial(
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
wg_gc.plot_klayout()
# -

scene = wg_gc.to_3d()
scene.show(show_ports=True)

# ### FabC
#
# Lets assume that fab C has similar technology to the generic PDK in gdsfactory and that you just want to remap some layers, and adjust the widths.
#

# +
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
xs_nc = partial(
    cross_section,
    width=WIDTH_NITRIDE_CBAND,
    layer=LAYER.WGN,
    bbox_layers=bbox_layers,
    bbox_offsets=bbox_offsets,
    add_pins=add_pins,
)
# Nitride Oband
xs_no = partial(
    cross_section,
    width=WIDTH_NITRIDE_OBAND,
    layer=LAYER.WGN,
    bbox_layers=bbox_layers,
    bbox_offsets=bbox_offsets,
    add_pins=add_pins,
)


cross_sections = dict(xs_nc=xs_nc, xs_no=xs_no, strip=xs_nc)

# LEAF cells have pins
mmi1x2_nc = partial(
    gf.components.mmi1x2,
    width=WIDTH_NITRIDE_CBAND,
    cross_section=xs_nc,
)
mmi1x2_no = partial(
    gf.components.mmi1x2,
    width=WIDTH_NITRIDE_OBAND,
    cross_section=xs_no,
)
bend_euler_nc = partial(
    gf.components.bend_euler,
    cross_section=xs_nc,
)
straight_nc = partial(
    gf.components.straight,
    cross_section=xs_nc,
)
bend_euler_no = partial(
    gf.components.bend_euler,
    cross_section=xs_no,
)
straight_no = partial(
    gf.components.straight,
    cross_section=xs_no,
)

gc_nc = partial(
    gf.components.grating_coupler_elliptical_te,
    grating_line_width=0.6,
    layer=LAYER.WGN,
    cross_section=xs_nc,
)

# HIERARCHICAL cells are made of leaf cells
mzi_nc = partial(
    gf.components.mzi,
    cross_section=xs_nc,
    splitter=mmi1x2_nc,
    straight=straight_nc,
    bend=bend_euler_nc,
)
mzi_no = partial(
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
# -


LAYER_VIEWS.layer_map.values()

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

c = mzi_gc.to_3d()
c.show(show_ports=True)

ls = get_layer_stack_fab_c()
