import gdsfactory as gf

c = gf.Component("pads_with_routes_with_wire_corners_no_orientation")  # from docs
pt = c << gf.components.pad()
pt.rotate(90)
pt.movey(+300)

pb = c << gf.components.pad()
# pb.movey(-100)
print(pb["pad"].orientation)

route = gf.routing.get_route_electrical(
    pt.ports["pad"], pb.ports["pad"], bend="wire_corner"
)
c.add(route.references)
c.show()
