"""Routes bundles of ports (river routing).

get bundle is the generic river routing function
route_bundle calls different function depending on the port orientation.

 - route_bundle_same_axis: ports facing each other with arbitrary pitch on each side
 - route_bundle_corner: 90Deg / 270Deg between ports with arbitrary pitch
 - route_bundle_udirect: ports with direct U-turns
 - route_bundle_uindirect: ports with indirect U-turns

"""

from __future__ import annotations

from functools import partial

import kfactory as kf
from kfactory.routing.optical import OpticalManhattanRoute

import gdsfactory as gf
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.wire import wire_corner
from gdsfactory.port import Port
from gdsfactory.routing.sort_ports import get_port_x, get_port_y
from gdsfactory.typings import (
    Component,
    ComponentSpec,
    CrossSectionSpec,
    LayerSpecs,
)


def get_min_spacing(
    ports1: list[Port],
    ports2: list[Port],
    separation: float = 5.0,
    radius: float = 5.0,
    sort_ports: bool = True,
) -> float:
    """Returns the minimum amount of spacing in um required to create a fanout.

    Args:
        ports1: first list of ports.
        ports2: second list of ports.
        separation: minimum separation between two straights in um.
        radius: bend radius in um.
        sort_ports: sort the ports according to the axis.

    """
    axis = "X" if ports1[0].orientation in [0, 180] else "Y"
    j = 0
    min_j = 0
    max_j = 0
    if sort_ports:
        if axis in {"X", "x"}:
            sorted(ports1, key=get_port_y)
            sorted(ports2, key=get_port_y)
        else:
            sorted(ports1, key=get_port_x)
            sorted(ports2, key=get_port_x)

    for port1, port2 in zip(ports1, ports2):
        if axis in {"X", "x"}:
            x1 = get_port_y(ports1)
            x2 = get_port_y(port2)
        else:
            x1 = get_port_x(port1)
            x2 = get_port_x(port2)
        if x2 >= x1:
            j += 1
        else:
            j -= 1
        if j < min_j:
            min_j = j
        if j > max_j:
            max_j = j
    j = 0

    return (max_j - min_j) * separation + 2 * radius + 1.0


def route_bundle(
    component: Component,
    ports1: list[Port],
    ports2: list[Port],
    separation: float = 3.0,
    bend: ComponentSpec = "bend_euler",
    sort_ports: bool = False,
    cross_section: CrossSectionSpec = "strip",
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    min_straight_taper: float = 100,
    taper: ComponentSpec | None = None,
    port_type: str | None = None,
    collision_check_layers: LayerSpecs | None = None,
    on_collision: str | None = "show_error",
    bboxes: list[kf.kdb.Box] | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | list[float] | None = None,
    straight: ComponentSpec = straight_function,
) -> list[OpticalManhattanRoute]:
    """Places a bundle of routes to connect two groups of ports.

    Routes connect a bundle of ports with a river router.
    Chooses the correct routing function depending on port angles.

    Args:
        component: component to add the routes to.
        ports1: list of starting ports.
        ports2: list of end ports.
        separation: bundle separation (center to center). Defaults to cross_section.width + cross_section.gap
        bend: function for the bend. Defaults to euler.
        sort_ports: sort port coordinates.
        cross_section: CrossSection or function that returns a cross_section.
        start_straight_length: straight length at the beginning of the route. If None, uses default value for the routing CrossSection.
        end_straight_length: end length at the beginning of the route. If None, uses default value for the routing CrossSection.
        min_straight_taper: minimum length for tapering the straight sections.
        taper: function for the taper. Defaults to None.
        port_type: type of port to place. Defaults to optical.
        collision_check_layers: list of layers to check for collisions.
        on_collision: action to take on collision. Defaults to show_error.
        bboxes: list of bounding boxes to avoid collisions.
        allow_width_mismatch: allow different port widths.
        radius: bend radius. If None, defaults to cross_section.radius.
        route_width: width of the route. If None, defaults to cross_section.width.
        straight: function for the straight. Defaults to straight.


    .. plot::
        :include-source:

        import gdsfactory as gf

        dy = 200.0
        xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

        pitch = 10.0
        N = len(xs1)
        xs2 = [-20 + i * pitch for i in range(N // 2)]
        xs2 += [400 + i * pitch for i in range(N // 2)]

        a1 = 90
        a2 = a1 + 180

        ports1 = [gf.Port(f"top_{i}", center=(xs1[i], +0), width=0.5, orientation=a1, layer=(1,0)) for i in range(N)]
        ports2 = [gf.Port(f"bot_{i}", center=(xs2[i], dy), width=0.5, orientation=a2, layer=(1,0)) for i in range(N)]

        c = gf.Component()
        gf.routing.route_bundle(c, ports1, ports2)
        c.plot()

    """
    # convert single port to list
    if isinstance(ports1, Port):
        ports1 = [ports1]

    if isinstance(ports2, Port):
        ports2 = [ports2]

    # convert ports dict to list
    if isinstance(ports1, dict):
        ports1 = list(ports1.values())

    if isinstance(ports2, dict):
        ports2 = list(ports2.values())

    ports1 = list(ports1)
    ports2 = list(ports2)

    port_type = port_type or ports1[0].port_type

    dbu = component.kcl.dbu

    if route_width and not isinstance(route_width, int | float):
        route_width = [width * dbu for width in route_width]

    if len(ports1) != len(ports2):
        raise ValueError(f"ports1={len(ports1)} and ports2={len(ports2)} must be equal")

    xs = gf.get_cross_section(cross_section)
    width = xs.width
    radius = radius or xs.radius
    width_dbu = round(width / component.kcl.dbu)
    taper_cell = gf.get_component(taper) if taper else None
    bend90 = (
        bend
        if isinstance(bend, Component)
        else gf.get_component(bend, cross_section=cross_section, radius=radius)
    )

    def straight_dbu(
        length: int, width: int = width_dbu, cross_section=cross_section
    ) -> Component:
        return gf.get_component(
            straight,
            length=length * component.kcl.dbu,
            width=width * component.kcl.dbu,
            cross_section=cross_section,
        )

    dbu = component.kcl.dbu
    end_straight = round(end_straight_length / dbu)
    start_straight = round(start_straight_length / dbu)

    if collision_check_layers:
        collision_check_layers = [
            gf.get_layer(layer) for layer in collision_check_layers
        ]

    return kf.routing.optical.route_bundle(
        component,
        ports1,
        ports2,
        round(separation / component.kcl.dbu),
        straight_factory=straight_dbu,
        bend90_cell=bend90,
        taper_cell=taper_cell,
        start_straights=start_straight,
        end_straights=end_straight,
        min_straight_taper=round(min_straight_taper / dbu),
        place_port_type=port_type,
        collision_check_layers=collision_check_layers,
        on_collision=on_collision,
        allow_width_mismatch=allow_width_mismatch,
        bboxes=bboxes or [],
        route_width=width_dbu,
        sort_ports=sort_ports,
    )


route_bundle_electrical = partial(
    route_bundle,
    bend=wire_corner,
    cross_section="metal_routing",
    allow_width_mismatch=True,
)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    columns = 2
    ptop = c << gf.components.pad_array(columns=columns, port_orientation=270)
    pbot = c << gf.components.pad_array(port_orientation=270, columns=columns)
    # pbot = c << gf.components.pad_array(port_orientation=90, columns=columns)

    ptop.dmovex(300)
    ptop.dmovey(300)
    routes = gf.routing.route_bundle_electrical(
        c,
        reversed(pbot.ports),
        ptop.ports,
        # end_straight_length=50,
        start_straight_length=100,
        separation=20,
        bboxes=[ptop.bbox(), pbot.bbox()],
    )

    c.show()
    # pbot.ports.print()

    # c = gf.Component("demo")
    # c1 = c << gf.components.mmi2x2()
    # c2 = c << gf.components.mmi2x2()
    # c2.dmove((100, 70))
    # routes = route_bundle(
    #     c,
    #     [c1.ports["o2"], c1.ports["o1"]],
    #     [c2.ports["o2"], c2.ports["o1"]],
    #     separation=5,
    #     cross_section="strip",
    #     # end_straight_length=0,
    #     # collision_check_layers=[(1, 0)],
    #     # bboxes=[c1.bbox(), c2.bbox()],
    #     # layer=(2, 0),
    #     # straight=partial(gf.components.straight, layer=(2, 0), width=1),
    # )
    # c.show()

    # dy = 200.0
    # xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]
    # pitch = 10.0
    # N = len(xs1)
    # xs2 = [-20 + i * pitch for i in range(N // 2)]
    # xs2 += [400 + i * pitch for i in range(N // 2)]
    # a1 = 90
    # a2 = a1 + 180

    # ports1 = [
    #     gf.Port(
    #         f"bot_{i}", center=(xs1[i], +0), width=0.5, orientation=a1, layer=(1, 0)
    #     )
    #     for i in range(N)
    # ]
    # ports2 = [
    #     gf.Port(
    #         f"top_{i}", center=(xs2[i], dy), width=0.5, orientation=a2, layer=(1, 0)
    #     )
    #     for i in range(N)
    # ]

    # c = gf.Component()
    # route_bundle(
    #     c,
    #     ports1,
    #     ports2,
    #     end_straight_length=1,
    #     start_straight_length=100,
    # )
    # c.add_ports(ports1)
    # c.add_ports(ports2)
    # c.show()
