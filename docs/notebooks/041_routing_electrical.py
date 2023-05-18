# # Routing electrical
#
# For routing low speed DC electrical ports you can use sharp corners instead of smooth bends.
#
# You can also define `port.orientation = None` to ignore the port orientation for low speed DC ports.

# ## get_route
#
# For single route between ports you can use `get_route_electrical`
#
# ### get_route_electrical
#
#
# `get_route_electrical` has `bend = wire_corner` with a 90deg bend corner.

# +
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

c = gf.Component("pads")
pt = c << gf.components.pad_array(orientation=270, columns=3)
pb = c << gf.components.pad_array(orientation=90, columns=3)
pt.move((70, 200))
c
# -

c = gf.Component("pads_with_routes_with_bends")
pt = c << gf.components.pad_array(orientation=270, columns=3)
pb = c << gf.components.pad_array(orientation=90, columns=3)
pt.move((70, 200))
route = gf.routing.get_route_electrical(
    pt.ports["e11"], pb.ports["e11"], bend="bend_euler", radius=30
)
c.add(route.references)
c

c = gf.Component("pads_with_routes_with_wire_corners")
pt = c << gf.components.pad_array(orientation=270, columns=3)
pb = c << gf.components.pad_array(orientation=90, columns=3)
pt.move((70, 200))
route = gf.routing.get_route_electrical(
    pt.ports["e11"], pb.ports["e11"], bend="wire_corner"
)
c.add(route.references)
c

c = gf.Component("pads_with_routes_with_wire_corners_no_orientation")
pt = c << gf.components.pad_array(orientation=None, columns=3)
pb = c << gf.components.pad_array(orientation=None, columns=3)
pt.move((70, 200))
route = gf.routing.get_route_electrical(
    pt.ports["e11"], pb.ports["e11"], bend="wire_corner"
)
c.add(route.references)
c

# +
c = gf.Component("multi-layer")
columns = 2
ptop = c << gf.components.pad_array(columns=columns)
pbot = c << gf.components.pad_array(orientation=90, columns=columns)

ptop.movex(300)
ptop.movey(300)
route = gf.routing.get_route_electrical_multilayer(
    ptop.ports["e11"],
    pbot.ports["e11"],
    end_straight_length=100,
)
c.add(route.references)
c
# -

# There is also `bend = wire_corner45` for 45deg bend corner with parametrizable "radius":

c = gf.Component("pads_with_routes_with_wire_corner45")
pt = c << gf.components.pad_array(orientation=270, columns=1)
pb = c << gf.components.pad_array(orientation=90, columns=1)
pt.move((300, 300))
route = gf.routing.get_route_electrical(
    pt.ports["e11"], pb.ports["e11"], bend="wire_corner45", radius=30
)
c.add(route.references)
c

c = gf.Component("pads_with_routes_with_wire_corner45")
pt = c << gf.components.pad_array(orientation=270, columns=1)
pb = c << gf.components.pad_array(orientation=90, columns=1)
pt.move((300, 300))
route = gf.routing.get_route_electrical(
    pt.ports["e11"], pb.ports["e11"], bend="wire_corner45", radius=100
)
c.add(route.references)
c

# ### route_quad

c = gf.Component("pads_route_quad")
pt = c << gf.components.pad_array(orientation=270, columns=3)
pb = c << gf.components.pad_array(orientation=90, columns=3)
pt.move((100, 200))
route = c << gf.routing.route_quad(pt.ports["e11"], pb.ports["e11"], layer=(49, 0))
c

# ### get_route_from_steps

c = gf.Component("pads_route_from_steps")
pt = c << gf.components.pad_array(orientation=270, columns=3)
pb = c << gf.components.pad_array(orientation=90, columns=3)
pt.move((100, 200))
route = gf.routing.get_route_from_steps(
    pb.ports["e11"],
    pt.ports["e11"],
    steps=[
        {"y": 200},
    ],
    cross_section="metal_routing",
    bend=gf.components.wire_corner,
)
c.add(route.references)
c

c = gf.Component("pads_route_from_steps_None_orientation")
pt = c << gf.components.pad_array(orientation=None, columns=3)
pb = c << gf.components.pad_array(orientation=None, columns=3)
pt.move((100, 200))
route = gf.routing.get_route_from_steps(
    pb.ports["e11"],
    pt.ports["e11"],
    steps=[
        {"y": 200},
    ],
    cross_section="metal_routing",
    bend=gf.components.wire_corner,
)
c.add(route.references)
c

# ## get_bundle
#
# ### get_bundle_electrical
#
# For routing groups of ports you can use `get_bundle` that returns a bundle of routes using a bundle router (also known as bus or river router)

# +
c = gf.Component("pads_bundle")
pt = c << gf.components.pad_array(orientation=270, columns=3)
pb = c << gf.components.pad_array(orientation=90, columns=3)
pt.move((100, 200))

routes = gf.routing.get_bundle_electrical(
    pb.ports, pt.ports, end_straight_length=60, separation=30
)

for route in routes:
    c.add(route.references)
c
# -

# ### get_bundle_from_steps_electrical

# +
c = gf.Component("pads_bundle_steps")
pt = c << gf.components.pad_array(
    gf.partial(gf.components.pad, size=(30, 30)),
    orientation=270,
    columns=3,
    spacing=(50, 0),
)
pb = c << gf.components.pad_array(orientation=90, columns=3)
pt.move((300, 500))

routes = gf.routing.get_bundle_from_steps_electrical(
    pb.ports, pt.ports, end_straight_length=60, separation=30, steps=[{"dy": 100}]
)

for route in routes:
    c.add(route.references)

c
# -

# ### get_bundle_electrical_multilayer
#
# To avoid metal crossings you can use one metal layer.

# +
c = gf.Component("get_bundle_multi_layer")
columns = 2
ptop = c << gf.components.pad_array(columns=columns)
pbot = c << gf.components.pad_array(orientation=90, columns=columns)

ptop.movex(300)
ptop.movey(300)
routes = gf.routing.get_bundle_electrical_multilayer(
    ptop.ports, pbot.ports, end_straight_length=100, separation=20
)
for route in routes:
    c.add(route.references)
c
# -

# ## Routing to pads
#
# You can also route to electrical pads.

c = gf.components.pad()
cc = gf.routing.add_pads_bot(component=c, port_names=("e1", "e4"), fanout_length=50)
cc

c = gf.components.straight_heater_metal(length=100.0)
cc = gf.routing.add_pads_top(component=c)
cc


c = gf.components.straight_heater_metal(length=100.0)
cc = gf.routing.add_pads_top(component=c, port_names=("e1",))
cc

n = west = north = south = east = 10
spacing = 20
c = gf.components.nxn(
    xsize=n * spacing,
    ysize=n * spacing,
    west=west,
    east=east,
    north=north,
    south=south,
    port_type="electrical",
    wg_width=10,
)
c

cc = gf.routing.add_pads_top(component=c)
cc
