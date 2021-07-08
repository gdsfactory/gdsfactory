"""
This is not a good solution either

"""


import pp

c = pp.Component("mzi_with_pads_sample_with_corners")
mzi = pp.components.mzi2x2(with_elec_connections=True)
pads = pp.components.pad_array(n=3, port_list=["S"])
p = c << pads
mzir = c << mzi
p.move((-150, 150))


for port1, port2 in zip(p.get_ports_list(), mzir.get_ports_list(port_type="dc")):
    route = pp.routing.get_route(port1, port2, waveguide="metal_routing", radius=10)
    c.add(route.references)
c.show()
