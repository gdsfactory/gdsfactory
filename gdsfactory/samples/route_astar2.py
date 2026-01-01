import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()


@gf.cell
def demo_route_astar_electrical() -> gf.Component:
    c = gf.Component()
    cross_section_name = "metal_routing"
    port_prefix = "e"
    bend = gf.components.wire_corner

    cross_section = gf.get_cross_section(cross_section_name)
    w = gf.components.straight(cross_section=cross_section)
    left = c << w
    right = c << w
    right.rotate(90)  # Type: ignore[arg-type]
    right.move((168, 63))

    obstacle = gf.components.rectangle(size=(250, 3), layer="M3")
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle3 = c << obstacle
    obstacle4 = c << obstacle
    obstacle4.rotate(90)  # Type: ignore[arg-type]
    obstacle1.ymin = 50
    obstacle1.xmin = -10
    obstacle2.xmin = 35
    obstacle3.ymin = 42
    obstacle3.xmin = 72.23  # Type: ignore
    obstacle4.xmin = 200
    obstacle4.ymin = 55
    port1 = left.ports[f"{port_prefix}1"]
    port2 = right.ports[f"{port_prefix}2"]

    gf.routing.route_astar(
        component=c,
        port1=port1,
        port2=port2,
        cross_section=cross_section,
        resolution=10,
        distance=15,
        avoid_layers=("M3",),
        bend=bend,
    )
    return c


if __name__ == "__main__":
    c = demo_route_astar_electrical()
    c.show()
