"""FIXME."""

import gdsfactory as gf


c = gf.Component("get_route_astar")
cross_section = "metal1"
w = gf.components.straight(cross_section=cross_section)

left = c << w
right = c << w
left.movex(-40)
right.move((100, 80))

obstacle = gf.components.rectangle(size=(100, 10), layer="M2")
obstacle1 = c << obstacle
obstacle2 = c << obstacle
obstacle1.ymin = 40
obstacle2.xmin = 25

port1 = left.ports["e2"]
port2 = right.ports["e2"]

routes = gf.routing.get_route_astar(
    component=c,
    port1=port1,
    port2=port2,
    cross_section="metal1",
    resolution=10,
    distance=7,
)
c.add(routes.references)
c.show()
