"""
This is not a good solution
get routes is a temporary solution

"""


import pp

c = pp.Component("mzi_with_pads_sample_with_corners")
mzi = pp.components.mzi2x2(with_elec_connections=True)
pads = pp.components.pad_array(n=3, port_list=["S"])
p = c << pads
mzir = c << mzi
p.move((-150, 150))


routes = pp.routing.get_routes(
    ports2=p.ports,
    ports1=mzir.get_ports_list(port_type="dc"),
)
c.add(routes.references)
c.show()
