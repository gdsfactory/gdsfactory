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

    tl.dmove((0, 300))
    br.dmove((500, 0))
    tr.dmove((500, 500))

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

    tl.dmove((0, 300))
    br.dmove((500, 0))
    tr.dmove((500, 500))

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


if __name__ == "__main__":
    c = pads_shorted()
    c.show()
    gdspath = c.write_gds()

    # import kfactory as kf

    # lib = kf.kcell.KCLayout()
    # lib.read(filename=str(gdspath))
    # c = lib[0]

    # l2n = kf.kdb.LayoutToNetlist(c.begin_shapes_rec(0))
    # for l_idx in c.kcl.layer_indices():
    #     l2n.connect(l2n.make_layer(l_idx, f"layer{l_idx}"))
    # l2n.extract_netlist()
    # print(l2n.netlist().to_s())
