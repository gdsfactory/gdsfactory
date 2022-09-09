import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.routing.get_route_astar import get_route_astar


@cell
def test_astar_pass() -> Component:
    c = gf.Component()
    rect1 = c << gf.components.rectangle()
    rect2 = c << gf.components.rectangle()
    rect3 = c << gf.components.rectangle((2, 2))
    rect2.move(destination=(8, 4))
    rect3.move(destination=(5.5, 1.5))

    port1 = Port(
        "o1", 0, rect1.center + (0, 3), cross_section=gf.get_cross_section("strip")
    )
    port2 = port1.copy("o2")
    port2.orientation = 180
    port2.center = rect2.center + (0, -3)
    c.add_ports([port1, port2])
    route = get_route_astar(c, port1, port2, radius=0.5, width=0.5)
    c.add(route.references)
    return c


# @cell
# def test_astar_fail() -> Component:
#     c = gf.Component()
#     rect1 = c << gf.components.rectangle()
#     rect2 = c << gf.components.rectangle()
#     rect3 = c << gf.components.rectangle((2, 2))
#     rect2.move(destination=(8, 4))
#     rect3.move(destination=(5.5, 1.5))

#     port1 = Port(
#         "o1", 0, rect1.center + (0, 3), cross_section=gf.get_cross_section("strip")
#     )
#     port2 = port1.copy("o2")
#     port2.orientation = 180
#     port2.center = rect2.center + (0, -3)
#     c.add_ports([port1, port2])

#     route = get_route_astar(c, port1, port2, radius=0.9, width=0.5)
#     c.add(route.references)
#     return c


if __name__ == "__main__":
    c = test_astar_pass()
    # c = test_astar_fail()
    c.show(show_ports=True)
