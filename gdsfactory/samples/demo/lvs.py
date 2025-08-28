"""LVS demo."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import ComponentFactory


@gf.cell
def pads_correct(
    pad: ComponentFactory = gf.components.pad, cross_section: str = "metal3"
) -> gf.Component:
    """Returns 2 pads connected with metal wires."""
    c = gf.Component()

    xs = gf.get_cross_section(cross_section)
    assert xs.layer is not None
    layer = gf.get_layer(xs.layer)

    pad_c = gf.get_component(pad)
    tl = c << pad_c
    bl = c << pad_c

    tr = c << pad_c
    br = c << pad_c

    tl.move((0, 300))
    br.move((500, 0))
    tr.move((500, 500))

    c.add_label("tl", position=tl.center, layer=layer)
    c.add_label("tr", position=tr.center, layer=layer)
    c.add_label("br", position=br.center, layer=layer)
    c.add_label("bl", position=bl.center, layer=layer)

    ports1 = [bl.ports["e3"], tl.ports["e3"]]
    ports2 = [br.ports["e1"], tr.ports["e1"]]
    gf.routing.route_bundle_electrical(c, ports1, ports2, cross_section=cross_section)
    return c


@gf.cell
def pads_shorted(
    pad: ComponentFactory = gf.components.pad, cross_section: str = "metal3"
) -> gf.Component:
    """Returns 2 pads connected with metal wires."""
    c = gf.Component()
    pad_c = gf.components.pad()
    xs = gf.get_cross_section(cross_section)
    assert xs.layer is not None
    layer = gf.get_layer(xs.layer)

    tl = c << pad_c
    bl = c << pad_c

    tr = c << pad_c
    br = c << pad_c

    tl.move((0, 300))
    br.move((500, 0))
    tr.move((500, 500))

    c.add_label("tl", position=tl.center, layer=layer)
    c.add_label("tr", position=tr.center, layer=layer)
    c.add_label("br", position=br.center, layer=layer)
    c.add_label("bl", position=bl.center, layer=layer)

    ports1 = [bl.ports["e3"], tl.ports["e3"]]
    ports2 = [br.ports["e1"], tr.ports["e1"]]
    gf.routing.route_bundle_electrical(c, ports1, ports2, cross_section=cross_section)

    gf.routing.route_single_electrical(
        c, bl.ports["e2"], tl.ports["e4"], cross_section=cross_section
    )
    return c
