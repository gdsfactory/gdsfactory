"""LVS demo."""

from __future__ import annotations

import gdsfactory as gf


@gf.cell
def pads_with_routes(radius: float = 10):
    """Returns 2 pads connected with metal wires."""
    c = gf.Component()
    pad = gf.components.pad()

    tl = c << pad
    bl = c << pad

    tr = c << pad
    br = c << pad

    tl.move((0, 300))
    br.move((500, 0))
    tr.move((500, 500))

    ports1 = [bl.ports["e3"], tl.ports["e3"]]
    ports2 = [br.ports["e1"], tr.ports["e1"]]
    routes = gf.routing.get_bundle(ports1, ports2, cross_section="metal3")

    for route in routes:
        c.add(route.references)

    return c


if __name__ == "__main__":
    c = pads_with_routes(radius=100)
    c.show(show_ports=True)
    gdspath = c.write_gds()

    import kfactory as kf

    lib = kf.kcell.KCLayout()
    lib.read(filename=str(gdspath))
    c = lib[0]

    l2n = kf.kdb.LayoutToNetlist(c.begin_shapes_rec(0))
    for l_idx in c.kcl.layer_indices():
        l2n.connect(l2n.make_layer(l_idx, f"layer{l_idx}"))
    l2n.extract_netlist()
    print(l2n.netlist().to_s())
