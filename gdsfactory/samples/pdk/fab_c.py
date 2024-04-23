"""FabC example."""

from __future__ import annotations

import sys
from functools import partial

from pydantic import BaseModel

import gdsfactory as gf
from gdsfactory.add_pins import add_pins_inside1nm as _add_pins_inside1nm
from gdsfactory.cross_section import get_cross_sections, strip
from gdsfactory.get_factories import get_cells
from gdsfactory.port import select_ports
from gdsfactory.technology import LayerLevel, LayerStack
from gdsfactory.typings import Layer


class LayerMap(BaseModel):
    WG: Layer = (10, 1)
    WG_CLAD: Layer = (10, 2)
    WGN: Layer = (34, 0)
    WGN_CLAD: Layer = (36, 0)
    PIN: Layer = (1, 10)


LAYER = LayerMap()
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
                layer=(1, 0),
                zmin=0.0,
                thickness=0.22,
            ),
            wgn=LayerLevel(
                layer=LAYER.WGN,
                zmin=0.22 + 0.1,
                thickness=0.4,
            ),
        )
    )


# avoid registering the function add pins
_add_pins = partial(_add_pins_inside1nm, pin_length=0.5)

######################
# cross_sections
######################
strip_sc = partial(
    strip,
    width=WIDTH_SILICON_CBAND,
    layer=LAYER.WG,
    bbox_layers=[LAYER.WG_CLAD],
    bbox_offsets=[3],
)
strip_so = partial(
    strip_sc,
    width=WIDTH_SILICON_OBAND,
)

strip_nc = partial(
    strip,
    width=WIDTH_NITRIDE_CBAND,
    layer=LAYER.WGN,
    bbox_layers=[LAYER.WGN_CLAD],
    bbox_offsets=[3],
)
strip_no = partial(
    strip_nc,
    width=WIDTH_NITRIDE_OBAND,
)

xs_sc = strip_sc()
xs_so = strip_so()
xs_nc = strip_nc()
xs_no = strip_no()

######################
# LEAF COMPONENTS with pins
######################

# customize the cell decorator for this PDK
cell = partial(gf.cell, post_process=[_add_pins])


@cell
def straight_sc(cross_section=strip_nc, **kwargs):
    return gf.components.straight(cross_section=cross_section, **kwargs)


@cell
def straight_so(cross_section=strip_so, **kwargs):
    return gf.components.straight(cross_section=cross_section, **kwargs)


@cell
def straight_nc(cross_section=strip_nc, **kwargs):
    return gf.components.straight(cross_section=cross_section, **kwargs)


@cell
def straight_no(cross_section=strip_no, **kwargs):
    return gf.components.straight(cross_section=cross_section, **kwargs)


######################
# bends
######################


@cell
def bend_euler_sc(cross_section=strip_sc, **kwargs):
    return gf.components.bend_euler(cross_section=cross_section, **kwargs)


@cell
def bend_euler_so(cross_section=strip_so, **kwargs):
    return gf.components.bend_euler(cross_section=cross_section, **kwargs)


@cell
def bend_euler_nc(cross_section=strip_nc, **kwargs):
    return gf.components.bend_euler(cross_section=cross_section, **kwargs)


@cell
def bend_euler_no(cross_section=strip_no, **kwargs):
    return gf.components.bend_euler(cross_section=cross_section, **kwargs)


######################
# MMI
######################


@cell
def mmi1x2_sc(width_mmi=3, cross_section=strip_sc, **kwargs):
    return gf.components.mmi1x2(
        cross_section=cross_section, width_mmi=width_mmi, **kwargs
    )


@cell
def mmi1x2_so(width_mmi=3, cross_section=strip_so, **kwargs):
    return gf.components.mmi1x2(
        cross_section=cross_section, width_mmi=width_mmi, **kwargs
    )


@cell
def mmi1x2_nc(width_mmi=3, cross_section=strip_nc, **kwargs):
    return gf.components.mmi1x2(
        cross_section=cross_section, width_mmi=width_mmi, **kwargs
    )


@cell
def mmi1x2_no(width_mmi=3, cross_section=strip_no, **kwargs):
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
    cross_section=xs_nc,
)


@cell
def gc_sc(**kwargs):
    return _gc_nc(**kwargs)


######################
# HIERARCHICAL COMPONENTS made of leaf components
######################

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

######################
# PDK
######################
# register all cells in this file
cells = get_cells(sys.modules[__name__])
cross_sections = get_cross_sections(sys.modules[__name__])
layer_stack = get_layer_stack_fab_c()

pdk = gf.Pdk(
    name="fab_c_demopdk",
    cells=cells,
    cross_sections=cross_sections,
    layer_stack=layer_stack,
)


if __name__ == "__main__":
    # c2 = mmi1x2_nc()
    # d2 = c2.to_dict()

    # from jsondiff import diff

    # d = diff(d1, d2)
    # c.show(show_ports=True)

    c = mmi1x2_nc()
    c.show()
