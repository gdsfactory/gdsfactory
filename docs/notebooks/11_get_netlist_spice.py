# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Netlist extractor SPICE
#
# You can also extract the SPICE netlist using klayout
#

# +
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


# -

c = pads_with_routes(radius=100)
gdspath = c.write_gds()
c


# +
import kfactory as kf

lib = kf.kcell.KLib()
lib.read(filename=str(gdspath))
c = lib[0]

l2n = kf.kdb.LayoutToNetlist(c.begin_shapes_rec(0))
for l_idx in c.klib.layer_indices():
    l2n.connect(l2n.make_layer(l_idx, f"layer{l_idx}"))
l2n.extract_netlist()
print(l2n.netlist().to_s())
# -

l2n.write_l2n("netlist_pads_correct.l2n")


# +
@gf.cell
def pads_with_routes_shorted(radius: float = 10):
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

    route = gf.routing.get_route(bl.ports["e2"], tl.ports["e4"], cross_section="metal3")
    c.add(route.references)
    return c


c = pads_with_routes_shorted(cache=False)
gdspath = c.write_gds()
c


# +
import kfactory as kf

lib = kf.kcell.KLib()
lib.read(filename=str(gdspath))
c = lib[0]

l2n = kf.kdb.LayoutToNetlist(c.begin_shapes_rec(0))
for l_idx in c.klib.layer_indices():
    l2n.connect(l2n.make_layer(l_idx, f"layer{l_idx}"))
l2n.extract_netlist()
print(l2n.netlist().to_s())
# -
