import gdsfactory as gf

c = gf.Component("get_route_astar")
w = gf.components.straight()
left = c << w
right = c << w
right.move((100, 80))

obstacle = gf.components.rectangle(size=(100, 10))
obstacle1 = c << obstacle
obstacle2 = c << obstacle
obstacle1.ymin = 40
obstacle2.xmin = 25

port1 = left.ports["o2"]
port2 = right.ports["o2"]

routes = gf.routing.get_route_astar(
    component=c,
    port1=port1,
    port2=port2,
    resolution=5,
    distance=5.5,
    radius=5,
)
c.add(routes.references)
c.show()
