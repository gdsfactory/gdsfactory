"""FabC example."""

from __future__ import annotations

import sys
from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.cross_section import get_cross_sections, strip
from gdsfactory.get_factories import get_cells
from gdsfactory.port import select_ports
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer
from gdsfactory.typings import Layer


class LAYER(gf.LayerEnum):
    layout = gf.constant(gf.kcl.layout)

    WG: Layer = (10, 1)
    WG_CLAD: Layer = (10, 2)
    WGN: Layer = (34, 0)
    WGN_CLAD: Layer = (36, 0)
    PIN: Layer = (1, 10)


WIDTH_SILICON_CBAND = 0.5
WIDTH_SILICON_OBAND = 0.4

WIDTH_NITRIDE_OBAND = 0.9
WIDTH_NITRIDE_CBAND = 1.0

select_ports_optical = partial(select_ports, layers_excluded=((100, 0),))


def get_layer_stack_fab_c(thickness: float = 350.0) -> LayerStack:
    """Returns generic LayerStack."""
    return LayerStack(
        layers=dict(
            wg=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WG),
                zmin=0.0,
                thickness=0.22,
            ),
            wgn=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WGN),
                zmin=0.22 + 0.1,
                thickness=0.4,
            ),
        )
    )


# avoid registering the function add pins using _underscore
_add_pins = partial(gf.add_pins.add_pins_inside1nm, pin_length=0.5, layer=LAYER.PIN)

######################
# cross_sections
######################
bbox_layers = (LAYER.WG_CLAD,)
bbox_offsets = (3,)

strip_sc = partial(
    strip,
    width=WIDTH_SILICON_CBAND,
    layer=LAYER.WG,
    bbox_layers=bbox_layers,
    bbox_offsets=bbox_offsets,
)
strip_so = partial(
    strip_sc,
    width=WIDTH_SILICON_OBAND,
)

strip_nc = partial(
    strip,
    width=WIDTH_NITRIDE_CBAND,
    layer=LAYER.WGN,
    bbox_layers=bbox_layers,
    bbox_offsets=bbox_offsets,
)
strip_no = partial(
    strip_nc,
    width=WIDTH_NITRIDE_OBAND,
)

strip = strip_sc()  # type: ignore[assignment]
xs_so = strip_so()
xs_nc = strip_nc()
xs_no = strip_no()

######################
# LEAF COMPONENTS with pins
######################


# customize the cell decorator for this PDK
_cell = gf.cell(post_process=(_add_pins,), info=dict(pdk="fab_c"))


@_cell
def straight_sc(cross_section: str = "strip_nc", **kwargs: Any) -> gf.Component:
    return gf.components.straight(cross_section=cross_section, **kwargs)


@_cell
def straight_so(cross_section: str = "strip_so", **kwargs: Any) -> gf.Component:
    return gf.components.straight(cross_section=cross_section, **kwargs)


@_cell
def straight_nc(cross_section: str = "strip_nc", **kwargs: Any) -> gf.Component:
    return gf.components.straight(cross_section=cross_section, **kwargs)


@_cell
def straight_no(cross_section: str = "strip_no", **kwargs: Any) -> gf.Component:
    return gf.components.straight(cross_section=cross_section, **kwargs)


######################
# bends
######################


@_cell
def bend_euler_sc(cross_section: str = "strip_sc", **kwargs: Any) -> gf.Component:
    return gf.components.bend_euler(cross_section=cross_section, **kwargs)


@_cell
def bend_euler_so(cross_section: str = "strip_so", **kwargs: Any) -> gf.Component:
    return gf.components.bend_euler(cross_section=cross_section, **kwargs)


@_cell
def bend_euler_nc(cross_section: str = "strip_nc", **kwargs: Any) -> gf.Component:
    return gf.components.bend_euler(cross_section=cross_section, **kwargs)


@_cell
def bend_euler_no(cross_section: str = "strip_no", **kwargs: Any) -> gf.Component:
    return gf.components.bend_euler(cross_section=cross_section, **kwargs)


######################
# MMI
######################


@_cell
def mmi1x2_sc(
    width_mmi: float = 3, cross_section: str = "strip_sc", **kwargs: Any
) -> gf.Component:
    return gf.components.mmi1x2(
        cross_section=cross_section, width_mmi=width_mmi, **kwargs
    )


@_cell
def mmi1x2_so(
    width_mmi: float = 3, cross_section: str = "strip_so", **kwargs: Any
) -> gf.Component:
    return gf.components.mmi1x2(
        cross_section=cross_section, width_mmi=width_mmi, **kwargs
    )


@_cell
def mmi1x2_nc(
    width_mmi: float = 3, cross_section: str = "strip_nc", **kwargs: Any
) -> gf.Component:
    return gf.components.mmi1x2(
        cross_section=cross_section, width_mmi=width_mmi, **kwargs
    )


@_cell
def mmi1x2_no(
    width_mmi: float = 3, cross_section: str = "strip_no", **kwargs: Any
) -> gf.Component:
    return gf.components.mmi1x2(
        cross_section=cross_section, width_mmi=width_mmi, **kwargs
    )


######################
# Grating couplers
######################
_gc_nc = partial(
    gf.components.grating_coupler_elliptical,
    grating_line_width=0.6,
    layer_slab=None,
    cross_section="strip_nc",
)


@_cell
def gc_sc(**kwargs: Any) -> gf.Component:
    return _gc_nc(**kwargs)


######################
# HIERARCHICAL COMPONENTS made of leaf components
######################

mzi_nc = partial(
    gf.components.mzi,  # type: ignore[has-type]
    splitter=mmi1x2_nc,
    straight=straight_nc,
    bend=bend_euler_nc,
    cross_section="strip_nc",
)
mzi_no = partial(
    gf.components.mzi,  # type: ignore[has-type]
    splitter=mmi1x2_no,
    straight=straight_no,
    bend=bend_euler_no,
    cross_section="strip_no",
)

######################
# PDK
######################
# register all cells in this file
cells = get_cells(sys.modules[__name__])
cross_sections = get_cross_sections(sys.modules[__name__])
layer_stack = get_layer_stack_fab_c()

PDK = gf.Pdk(
    name="fab_c_demopdk",
    cells=cells,
    cross_sections=cross_sections,
    layer_stack=layer_stack,
    layers=LAYER,
)


if __name__ == "__main__":
    # c2 = mmi1x2_nc()
    # d2 = c2.to_dict()

    # from jsondiff import diff

    # d = diff(d1, d2)
    # c.show()

    # c = straight_nc()
    c = mzi_nc(length_x=100)
    # _add_pins(c)
    # gf.add_pins.add_pins(c)
    # c = mmi1x2_sc()
    # c.pprint_ports()
    c.show()
