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
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.via_corner import via_corner
from gdsfactory.components.wire import wire_corner
from gdsfactory.port import Port
from gdsfactory.routing.sort_ports import get_port_x, get_port_y
from gdsfactory.routing.sort_ports import sort_ports as sort_ports_function
from gdsfactory.routing.validation import (
    is_invalid_bundle_topology,
    make_error_traces,
)
from gdsfactory.typings import (
    Component,
    ComponentSpec,
    CrossSectionSpec,
    MultiCrossSectionAngleSpec,
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
    straight: ComponentSpec = straight_function,
    bend: ComponentSpec = bend_euler,
    sort_ports: bool = False,
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = "xs_sc",
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    enforce_port_ordering: bool = True,
    min_straight_taper: float = 100,
    taper: ComponentSpec | None = None,
    **kwargs,
) -> list[OpticalManhattanRoute]:
    """Places a bundle of routes to connect two groups of ports.

    Routes connect a bundle of ports with a river router.
    Chooses the correct routing function depending on port angles.

    Args:
        component: component to add the routes to.
        ports1: list of starting ports.
        ports2: list of end ports.
        separation: bundle separation (center to center). Defaults to cross_section.width + cross_section.gap
        extension_length: adds straight extension.
        bend: function for the bend. Defaults to euler.
        sort_ports: sort port coordinates.
        cross_section: CrossSection or function that returns a cross_section.
        start_straight_length: straight length at the beginning of the route. If None, uses default value for the routing CrossSection.
        end_straight_length: end length at the beginning of the route. If None, uses default value for the routing CrossSection.
        path_length_match_loops: Integer number of loops to add to bundle \
                for path length matching. Path-length matching won't be attempted if this is set to None.
        path_length_match_extra_length: Extra length to add to path length matching loops \
                (requires path_length_match_loops != None).
        path_length_match_modify_segment_i: Index of straight segment to add path length matching loops to \
                (requires path_length_match_loops != None).
        enforce_port_ordering: If True, enforce that the ports are connected in the specific order.
        steps: specify waypoint steps to route using route_bundle_from_steps.
        waypoints: specify waypoints to route using route_bundle_from_steps.

    Keyword Args:
        width: main layer waveguide width (um).
        layer: main layer for waveguide.
        width_wide: wide waveguides width (um) for low loss routing.
        auto_widen: taper to wide waveguides for low loss routing.
        auto_widen_minimum_length: minimum straight length for auto_widen.
        taper_length: taper_length for auto_widen.
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        cladding_layers: list of layers to extrude.
        cladding_offsets: list of offset from main Section edge.
        radius: bend radius (um).
        sections: list of Sections(width, offset, layer, ports).
        port_names: for input and output ('o1', 'o2').
        port_types: for input and output: electrical, optical, vertical_te ...
        min_length: defaults to 1nm = 10e-3um for routing.
        snap_to_grid: can snap points to grid when extruding the path.

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

    if isinstance(cross_section, list | tuple):
        xs_list = []
        for element in cross_section:
            xs, angles = element
            xs = gf.get_cross_section(xs)
            xs = xs.copy(**kwargs)  # Shallow copy
            xs_list.append((xs, angles))
        cross_section = xs_list

    else:
        cross_section = gf.get_cross_section(cross_section)
        cross_section = cross_section.copy(**kwargs)

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

    if len(ports1) != len(ports2):
        raise ValueError(f"ports1={len(ports1)} and ports2={len(ports2)} must be equal")

    start_port_angles = {p.orientation for p in ports1}
    if len(start_port_angles) > 1:
        raise ValueError(f"All start port angles {start_port_angles} must be equal")

    if sort_ports:
        ports1, ports2 = sort_ports_function(
            ports1, ports2, enforce_port_ordering=enforce_port_ordering
        )
    if enforce_port_ordering and is_invalid_bundle_topology(ports1, ports2):
        return make_error_traces(
            component,
            ports1,
            ports2,
            "Not a routable bundle topology! Try flipping the port order of either ports1 or ports2",
        )

    xs = gf.get_cross_section(cross_section, **kwargs)
    width = xs.width
    width_dbu = width / component.kcl.dbu
    straight = partial(straight, width=width, cross_section=cross_section)
    taper_cell = taper(cross_section=cross_section) if taper else None
    bend90 = (
        gf.get_component(bend, cross_section=xs)
        if not isinstance(bend, Component)
        else bend
    )

    def straight_dbu(
        length: int, width: int = width_dbu, cross_section=cross_section, **kwargs
    ) -> Component:
        return straight(
            length=length * component.kcl.dbu,
            width=width * component.kcl.dbu,
            cross_section=cross_section,
            **kwargs,
        )

    dbu = component.kcl.dbu
    end_straight = round(end_straight_length / dbu)
    start_straight = round(start_straight_length / dbu)

    kf.routing.optical.route_bundle(
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
    )


route_bundle_electrical = partial(
    route_bundle, bend=wire_corner, cross_section="xs_metal_routing"
)

route_bundle_electrical_multilayer = partial(
    route_bundle,
    bend=via_corner,
    cross_section=[
        (gf.cross_section.metal2, (90, 270)),
        ("xs_metal_routing", (0, 180)),
    ],
)


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.Component("route_bundle_multi_layer")
    # columns = 2
    # ptop = c << gf.components.pad_array(columns=columns)
    # pbot = c << gf.components.pad_array(orientation=90, columns=columns)

    # ptop.movex(300)
    # ptop.movey(300)
    # routes = gf.routing.route_bundle_electrical_multilayer(
    #     ptop.ports, pbot.ports, end_straight_length=100, separation=20
    # )
    # for route in routes:
    #     c.add(route.references)

    # c.show()

    c = gf.Component("demo")
    c1 = c << gf.components.mmi2x2()
    c2 = c << gf.components.mmi2x2()
    c2.d.move((100, 70))
    routes = route_bundle(
        c,
        [c2.ports["o2"], c2.ports["o1"]],
        [c1.ports["o1"], c1.ports["o2"]],
        enforce_port_ordering=False,
        separation=5,
        cross_section="xs_sc",
        end_straight_length=0,
        # layer=(2, 0),
        # straight=partial(gf.components.straight, layer=(2, 0), width=1),
    )
    c.show()

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
