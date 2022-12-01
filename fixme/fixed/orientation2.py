import gdsfactory as gf

c = gf.Component("pads_with_routes_with_wire_corners_no_orientation")  # from docs
pt = c << gf.components.pad_array(orientation=None, columns=3)
pb = c << gf.components.pad_array(orientation=None, columns=3)

pt.rotate(90)
pt.movey(300)
pt.movex(-300)

route = gf.routing.get_route_electrical(
    pt.ports["e11"], pb.ports["e11"], bend="wire_corner"
)
c.add(route.references)
c.show()
