import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()
    cross_section_name = "strip"
    port_prefix = "o"
    bend = gf.components.bend_euler

    cross_section = gf.get_cross_section(cross_section_name, radius=5)
    w = gf.components.straight(cross_section=cross_section)
    left = c << w
    right = c << w
    right.rotate(90)
    right.move((168, 63))

    obstacle = gf.components.rectangle(size=(250, 3), layer="M2")
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle3 = c << obstacle
    obstacle4 = c << obstacle
    obstacle4.rotate(90)
    obstacle1.ymin = 50
    obstacle1.xmin = -10
    obstacle2.xmin = 35
    obstacle3.ymin = 42
    obstacle3.xmin = 72.23
    obstacle4.xmin = 200
    obstacle4.ymin = 55
    port1 = left.ports[f"{port_prefix}1"]
    port2 = right.ports[f"{port_prefix}2"]

    # This gdsfactory function calculates a route for a waveguide using the A* algorithm,
    # which is a popular pathfinding algorithm known for finding the shortest path between two points on a grid.
    route = gf.routing.route_astar(
        component=c,
        port1=port1,
        port2=port2,
        cross_section=cross_section,
        # resolution=15: This parameter controls the grid size for the A* search algorithm. The router converts the component's layout into a grid,
        # and a value of 15 means the grid cells are 15 nanometers on each side (since gdsfactory's default unit is micrometers, this corresponds to 0.015 um).
        # A smaller resolution creates a finer grid, which can find more complex paths but takes longer to compute.
        resolution=15,
        distance=12,
        avoid_layers=("M2",),
        bend=bend,
    )
    c.show()
