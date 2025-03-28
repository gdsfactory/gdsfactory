"""Routes bundles of ports (river routing).

get bundle is the generic river routing function
route_bundle calls different function depending on the port orientation.

 - route_bundle_same_axis: ports facing each other with arbitrary pitch on each side
 - route_bundle_corner: 90Deg / 270Deg between ports with arbitrary pitch
 - route_bundle_udirect: ports with direct U-turns
 - route_bundle_uindirect: ports with indirect U-turns

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Literal

import kfactory as kf
from kfactory.routing.generic import ManhattanRoute

import gdsfactory as gf
from gdsfactory.routing.auto_taper import add_auto_tapers
from gdsfactory.routing.sort_ports import get_port_x, get_port_y
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    ComponentSpec,
    Coordinates,
    CrossSectionSpec,
    LayerSpec,
    LayerSpecs,
    Ports,
)

OpticalManhattanRoute = ManhattanRoute

TOLERANCE = 1


def get_min_spacing(
    ports1: Ports,
    ports2: Ports,
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
            x1 = get_port_y(port1)
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
    component: gf.Component,
    ports1: Ports,
    ports2: Ports,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    separation: float = 3.0,
    bend: ComponentSpec = "bend_euler",
    sort_ports: bool = False,
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    min_straight_taper: float = 100,
    taper: ComponentSpec | None = None,
    port_type: str | None = None,
    collision_check_layers: LayerSpecs | None = None,
    on_collision: Literal["error", "show_error"] | None = None,
    bboxes: Sequence[kf.kdb.DBox] | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | None = None,
    straight: ComponentSpec = "straight",
    auto_taper: bool = True,
    waypoints: Coordinates | None = None,
    steps: Sequence[Mapping[str, int | float]] | None = None,
    start_angles: float | list[float] | None = None,
    end_angles: float | list[float] | None = None,
    router: Literal["optical", "electrical"] | None = None,
) -> list[ManhattanRoute]:
    """Places a bundle of routes to connect two groups of ports.

    Routes connect a bundle of ports with a river router.
    Chooses the correct routing function depending on port angles.

    Args:
        component: component to add the routes to.
        ports1: list of starting ports.
        ports2: list of end ports.
        cross_section: CrossSection or function that returns a cross_section.
        layer: layer to use for the route.
        separation: bundle separation (center to center). Defaults to cross_section.width + cross_section.gap
        bend: function for the bend. Defaults to euler.
        sort_ports: sort port coordinates.
        start_straight_length: straight length at the beginning of the route. If None, uses default value for the routing CrossSection.
        end_straight_length: end length at the beginning of the route. If None, uses default value for the routing CrossSection.
        min_straight_taper: minimum length for tapering the straight sections.
        taper: function for the taper. Defaults to None.
        port_type: type of port to place. Defaults to optical.
        collision_check_layers: list of layers to check for collisions.
        on_collision: action to take on collision. Defaults to None (ignore).
        bboxes: list of bounding boxes to avoid collisions.
        allow_width_mismatch: allow different port widths.
        radius: bend radius. If None, defaults to cross_section.radius.
        route_width: width of the route. If None, defaults to cross_section.width.
        straight: function for the straight. Defaults to straight.
        auto_taper: if True, auto-tapers ports to the cross-section of the route.
        waypoints: list of waypoints to add to the route.
        steps: list of steps to add to the route.
        start_angles: list of start angles for the routes. Only used for electrical ports.
        end_angles: list of end angles for the routes. Only used for electrical ports.
        router: Set the type of router to use, either the optical one or the electrical one.
            If None, the router is optical unless the port_type is "electrical".

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

        ports1 = [gf.Port(name=f"top_{i}", center=(xs1[i], +0), width=0.5, orientation=a1, layer=(1,0)) for i in range(N)]
        ports2 = [gf.Port(name=f"bot_{i}", center=(xs2[i], dy), width=0.5, orientation=a2, layer=(1,0)) for i in range(N)]

        c = gf.Component()
        gf.routing.route_bundle(component=c, ports1=ports1, ports2=ports2, cross_section='strip', separation=5)
        c.plot()

    """
    if layer and cross_section:
        raise ValueError(
            f"Cannot have both {layer=} and {cross_section=} provided. Choose one."
        )
    if cross_section is None:
        if layer is not None and route_width is not None:
            cross_section = partial(
                gf.cross_section.cross_section, layer=layer, width=route_width
            )

        else:
            raise ValueError(
                f"Either {cross_section=} or {layer=} and {route_width=} must be provided"
            )

    c = component
    ports1_ = list(ports1)
    ports2_ = list(ports2)
    port_type = port_type or ports1_[0].port_type

    if len(ports1_) != len(ports2_):
        raise ValueError(
            f"ports1={len(ports1_)} and ports2={len(ports2_)} must be equal"
        )
    if route_width is None or route_width == 0:
        xs = gf.get_cross_section(cross_section)
    else:
        xs = gf.get_cross_section(cross_section, width=route_width)
    width = route_width or xs.width

    radius = radius or xs.radius
    taper_cell = gf.get_component(taper) if taper else None

    if collision_check_layers:
        collision_check_layer_enums = [
            gf.get_layer(layer) for layer in collision_check_layers
        ]
    else:
        collision_check_layer_enums = None

    if auto_taper:
        ports1_ = add_auto_tapers(component, ports1_, cross_section)
        ports2_ = add_auto_tapers(component, ports2_, cross_section)

    if steps and waypoints:
        raise ValueError("Cannot have both steps and waypoints")

    if steps:
        waypoints = []
        x, y = ports1_[0].center
        for d in steps:
            if not STEP_DIRECTIVES.issuperset(d):
                invalid_step_directives = list(set(d.keys()) - STEP_DIRECTIVES)
                raise ValueError(
                    f"Invalid step directives: {invalid_step_directives}."
                    f"Valid directives are {list(STEP_DIRECTIVES)}"
                )
            x = d.get("x", x) + d.get("dx", 0)
            y = d.get("y", y) + d.get("dy", 0)
            waypoints += [(x, y)]
    if waypoints is not None and not isinstance(waypoints[0], kf.kdb.DPoint):
        waypoints_: list[kf.kdb.DPoint] | None = [
            kf.kdb.DPoint(p[0], p[1]) for p in waypoints
        ]
    else:
        waypoints_ = waypoints

    router = router or "electrical" if port_type == "electrical" else "optical"
    if router == "electrical":
        return kf.routing.electrical.route_bundle(
            component,
            ports1_,
            ports2_,
            separation=separation,
            starts=start_straight_length,
            ends=end_straight_length,
            collision_check_layers=[
                c.kcl.layout.get_info(layer) for layer in collision_check_layer_enums
            ]
            if collision_check_layer_enums is not None
            else None,
            on_collision=on_collision,
            bboxes=bboxes,
            route_width=width,
            sort_ports=sort_ports,
            waypoints=waypoints_,
            end_angles=end_angles,
            start_angles=start_angles,
        )

    bend90 = (
        bend
        if isinstance(bend, gf.Component)
        else gf.get_component(
            bend, cross_section=cross_section, radius=radius, width=width
        )
    )

    def straight_um(width: float, length: float) -> gf.Component:
        return gf.get_component(
            straight, length=length, cross_section=cross_section, width=width
        )

    return kf.routing.optical.route_bundle(
        component,
        ports1_,
        ports2_,
        separation=separation,
        straight_factory=straight_um,
        bend90_cell=bend90,
        taper_cell=taper_cell,
        starts=start_straight_length,
        ends=end_straight_length,
        min_straight_taper=min_straight_taper,
        place_port_type=port_type,
        collision_check_layers=[
            c.kcl.layout.get_info(layer) for layer in collision_check_layer_enums
        ]
        if collision_check_layer_enums
        else None,
        on_collision=on_collision,
        allow_width_mismatch=allow_width_mismatch,
        bboxes=list(bboxes or []),
        route_width=width,
        sort_ports=sort_ports,
        waypoints=waypoints_,
        end_angles=end_angles,
        start_angles=start_angles,
    )


route_bundle_electrical = partial(
    route_bundle,
    bend="wire_corner",
    allow_width_mismatch=True,
)


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.generic_tech import LAYER

    pdk = gf.get_active_pdk()
    pdk.layer_transitions[LAYER.WG] = partial(
        gf.c.taper, cross_section="rib", length=20
    )
    pdk.layer_transitions[LAYER.WG, LAYER.WGN] = gf.c.taper_sc_nc
    pdk.layer_transitions[LAYER.WGN, LAYER.WG] = gf.c.taper_nc_sc

    # c = gf.Component()
    # columns = 2
    # ptop = c << gf.components.pad_array(columns=columns, port_orientation=270)
    # pbot = c << gf.components.pad_array(port_orientation=270, columns=columns)
    # # pbot = c << gf.components.pad_array(port_orientation=90, columns=columns)

    # ptop.movex(300)
    # ptop.movey(300)
    # routes = gf.routing.route_bundle_electrical(
    #     c,
    #     list(reversed(pbot.ports)),
    #     ptop.ports,
    #     # end_straight_length=50,
    #     start_straight_length=100,
    #     separation=20,
    #     bboxes=[ptop.bbox(), pbot.bbox()],
    #     cross_section="metal_routing",
    #     start_angles=None,
    #     end_angles=None,
    #     route_width=2,
    #     steps=[
    #         {"dy": 1, "dx": 1},
    #         {"dy": 2, "dx": 1},
    #     ],
    # )

    # c.show()
    # pbot.ports.print()

    c = gf.Component(name="demo")
    c1 = c << gf.components.mmi2x2()
    c2 = c << gf.components.mmi2x2()
    c2.move((100, 70))
    routes = route_bundle(
        c,
        [c1.ports["o2"], c1.ports["o1"]],
        [c2.ports["o2"], c2.ports["o1"]],
        separation=5,
        cross_section="strip",
        sort_ports=True,
        # end_straight_length=0,
        # collision_check_layers=[(1, 0)],
        # bboxes=[c1.bbox(), c2.bbox()],
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

    # nitride case
    # c = gf.Component()
    # c1 = c << gf.components.straight(width=0.5, cross_section="strip")
    # c2 = c << gf.components.straight(cross_section="strip", width=0.5)
    # c2.move((150, 50))
    # routes = route_bundle(
    #     c,
    #     [c1.ports["o2"]],
    #     [c2.ports["o1"]],
    #     separation=5,
    #     cross_section="nitride",
    #     auto_taper=True,
    # )
    # c.show()

    # rib

    # c = gf.Component()
    # # c1 = c << gf.components.straight(width=2, cross_section="rib")
    # # c2 = c << gf.components.straight(cross_section="rib", width=1)
    # c1 = c << gf.components.straight(cross_section="rib", width=2)
    # c2 = c << gf.components.straight(cross_section="rib", width=4)
    # c2.move((300, 70))
    # routes = route_bundle(
    #     c,
    #     [c1.ports["o2"]],
    #     [c2.ports["o1"]],
    #     # waypoints=[(200, 40), (200, 50)],
    #     # steps=[dict(dx=50, dy=100)],
    #     steps=[dict(dx=50, dy=100), dict(dy=100)],
    #     separation=5,
    #     cross_section="rib",
    #     auto_taper=True,
    #     # taper=partial(gf.c.taper, cross_section="rib", length=20),
    #     # taper=gf.c.taper_sc_nc,
    #     # taper=gf.c.taper,
    # )
    # c.show()

    # c = gf.Component()
    # w = gf.components.array(gf.c.straight, columns=1, rows=3, spacing=(3, 3))
    # left = c << w
    # right = c << w
    # right.move((100, 80))

    # obstacle = gf.components.rectangle(size=(100, 10))
    # obstacle1 = c << obstacle
    # obstacle2 = c << obstacle
    # obstacle1.ymin = 40
    # obstacle2.xmin = 35

    # ports1 = left.ports.filter(orientation=0)
    # ports2 = right.ports.filter(orientation=180)

    # routes = gf.routing.route_bundle(
    #     c,
    #     ports1,
    #     ports2,
    #     steps=[
    #         {"dy": 30, "dx": 50},
    #         {"dy": 30, "dx": 50},
    #         # {"dx": 90},
    #     ],
    #     cross_section="strip",
    #     # layer=(2, 0),
    #     route_width=0.2,
    # )
    # c.show()
