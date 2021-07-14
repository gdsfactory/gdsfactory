"""Electrical routing creates some strange artifacts.

See for example the route to the middle pad in this example.

There is a more clear direct route.
"""

import pp

c = pp.Component("mzi_with_pads_sample_with_corners")
mzi = pp.components.mzi2x2(with_elec_connections=True)
pads = pp.components.pad_array(n=3, port_list=["S"])
p = c << pads
mzir = c << mzi
p.move((-150, 150))

routes = pp.routing.get_bundle(
    ports1=p.ports,
    ports2=mzir.get_ports_list(port_type="dc"),
    waveguide="metal_routing",
    bend_factory=pp.components.wire_corner,
)
for route in routes:
    c.add(route.references)
c.show()
