from __future__ import annotations

from typing import Any, Literal

import kfactory as kf
import numpy as np
from kfactory.routing.generic import ManhattanRoute

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import Port, flipped
from gdsfactory.routing.route_single import route_single


def sort_key_west_to_east(port: Port) -> float:
    return port.dx


def sort_key_east_to_west(port: Port) -> float:
    return -port.dx


def sort_key_south_to_north(port: Port) -> float:
    return port.dy


def sort_key_north_to_south(port: Port) -> float:
    return -port.dy


def route_ports_to_side(
    component: Component,
    ports: list[kf.Port] | Component | list[gf.Port] | None = None,
    side: Literal["north", "east", "south", "west"] = "north",
    x: float | None = None,
    y: float | None = None,
    **kwargs,
) -> tuple[list[ManhattanRoute], list[kf.Port]]:
    """Routes ports to a given side.

    Args:
        component: component to route.
        ports: list/dict/Component/ComponentReference to route to a side.
        side: 'north', 'south', 'east' or 'west'.
        x: position to route ports for east/west. None, uses most east/west value.
        y: position to route ports for south/north. None, uses most north/south value.
        kwargs: additional arguments to pass to the routing function.

    Keyword Args:
      radius: in um.
      separation: in um.
      extend_bottom/extend_top for east/west routing.
      extend_left, extend_right for south/north routing.

    Returns:
        List of routes: with routing elements.
        List of ports: of the new ports.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component('sample_route_sides')
        dummy = gf.components.nxn(north=2, south=2, west=2, east=2)
        sides = ["north", "south", "east", "west"]
        d = 100
        positions = [(0, 0), (d, 0), (d, d), (0, d)]

        for pos, side in zip(positions, sides):
            dummy_ref = dummy.ref(position=pos)
            c.add(dummy_ref)
            routes, ports = gf.routing.route_ports_to_side(dummy_ref, side, layer=(1, 0))
            for route in routes:
                c.add(route.references)
            for i, p in enumerate(ports):
                c.add_port(name=f"{side[0]}{i}", port=p)

        c.plot()

    """
    if not ports:
        return [], []

    if side in {"north", "south"}:
        func_route = route_ports_to_y
        xy = y if y is not None else side
    elif side in {"west", "east"}:
        xy = x if x is not None else side
        func_route = route_ports_to_x
    else:
        raise ValueError(f"side = {side} not valid (north, south, west, east)")

    ports = ports or component.ports
    return func_route(component, ports, xy, side=side, **kwargs)


def route_ports_to_north(list_ports, **kwargs):
    return route_ports_to_side(list_ports, side="north", **kwargs)


def route_ports_to_south(list_ports, **kwargs):
    return route_ports_to_side(list_ports, side="south", **kwargs)


def route_ports_to_west(list_ports, **kwargs):
    return route_ports_to_side(list_ports, side="west", **kwargs)


def route_ports_to_east(list_ports, **kwargs):
    return route_ports_to_side(list_ports, side="east", **kwargs)


def route_ports_to_x(
    component: Component,
    list_ports: list[Port],
    x: float | Literal["east", "west"] = "east",
    separation: float = 10.0,
    radius: float = 10.0,
    extend_bottom: float = 0.0,
    extend_top: float = 0.0,
    extension_length: float = 0.0,
    y0_bottom: float | None = None,
    y0_top: float | None = None,
    backward_port_side_split_index: int = 0,
    start_straight_length: float = 0.01,
    dx_start: float | None = None,
    dy_start: float | None = None,
    side: Literal["east", "west"] = "east",
    **routing_func_args,
) -> tuple[list[ManhattanRoute], list[kf.Port]]:
    """Returns route to x.

    Args:
        component: component to route.
        list_ports: reasonably well behaved list of ports.
           ports facing north ports are norther than any other ports
           ports facing south ports are souther ...
           ports facing west ports are the wester ...
           ports facing east ports are the easter ...
        x: float or string.
           if float: x coordinate to which the ports will be routed
           if string: "east" -> route to east
           if string: "west" -> route to west
        separation: in um.
        radius: in um.
        extend_bottom: in um.
        extend_top: in um.
        extension_length: in um.
        y0_bottom: in um.
        y0_top: in um.
        backward_port_side_split_index: integer represents and index in the list of backwards ports (bottom to top)
            all ports with an index strictly lower or equal are routed bottom
            all ports with an index larger or equal are routed top.
        start_straight_length: in um.
        dx_start: override minimum starting x distance.
        dy_start: override minimum starting y distance.
        side: "east" or "west".
        routing_func_args: additional arguments to pass to the routing function.

    Returns:
        routes: list of routes
        ports: list of the new optical ports

    1. routes the bottom-half of the ports facing opposite side of x
    2. routes the south ports
    3. front ports
    4. north ports

    """
    north_ports = [p for p in list_ports if p.orientation == 90]
    south_ports = [p for p in list_ports if p.orientation == 270]
    east_ports = [p for p in list_ports if p.orientation == 0]
    west_ports = [p for p in list_ports if p.orientation == 180]

    epsilon = 1.0
    a = epsilon + max(radius, separation)
    bx = epsilon + max(radius, dx_start) if dx_start else a
    by = epsilon + max(radius, dy_start) if dy_start else a

    xs = [p.dx for p in list_ports]
    ys = [p.dy for p in list_ports]

    if y0_bottom is None:
        y0_bottom = min(ys) - by

    y0_bottom -= extend_bottom

    if y0_top is None:
        y0_top = max(ys) + (max(radius, dy_start) if dy_start else a)
    y0_top += extend_top

    if x == "west" and extension_length > 0:
        extension_length = -extension_length

    if x == "east":
        x = max(p.dx for p in list_ports) + bx
    elif x == "west":
        x = min(p.dx for p in list_ports) - bx
    elif isinstance(x, float | int):
        pass
    else:
        raise ValueError(f"x={x!r} should be a float or east or west")

    if x < min(xs):
        sort_key_north = sort_key_west_to_east
        sort_key_south = sort_key_west_to_east
        forward_ports = west_ports
        backward_ports = east_ports
        angle = 0

    elif x > max(xs):
        sort_key_south = sort_key_east_to_west
        sort_key_north = sort_key_east_to_west
        forward_ports = east_ports
        backward_ports = west_ports
        angle = 180
    else:
        raise ValueError("x should be either to the east or to the west of all ports")

    # forward_ports.sort()
    north_ports.sort(key=sort_key_north)
    south_ports.sort(key=sort_key_south)
    forward_ports.sort(key=sort_key_south_to_north)

    backward_ports.sort(key=sort_key_south_to_north)
    backward_ports_thru_south = backward_ports[:backward_port_side_split_index]
    backward_ports_thru_north = backward_ports[backward_port_side_split_index:]
    backward_ports_thru_south.sort(key=sort_key_south_to_north)
    backward_ports_thru_north.sort(key=sort_key_north_to_south)

    routes = []
    ports = []

    def add_port(
        p, y, l_elements, l_ports, start_straight_length=start_straight_length
    ) -> None:
        if side == "west":
            angle = 0

        elif side == "east":
            angle = 180

        new_port = p.copy()
        new_port.orientation = angle
        new_port.dx = x + extension_length
        new_port.dy = y

        new_port2 = new_port.copy()
        new_port2.trans *= gf.kdb.Trans.R180

        l_elements += [
            route_single(
                component,
                p,
                new_port,
                start_straight_length=start_straight_length,
                radius=radius,
                **routing_func_args,
            )
        ]
        l_ports += [new_port2]

    y_optical_bot = y0_bottom
    for p in south_ports:
        add_port(p, y_optical_bot, routes, ports)
        y_optical_bot -= separation

    for p in forward_ports:
        add_port(p, p.dy, routes, ports)

    y_optical_top = y0_top
    for p in north_ports:
        add_port(p, y_optical_top, routes, ports)
        y_optical_top += separation

    start_straight_length_section = start_straight_length
    max_x = max(xs)
    min_x = min(xs)

    for p in backward_ports_thru_north:
        # Extend ports if necessary
        if angle == 0 and p.dx < max_x:
            start_straight_length_section = max_x - p.dx
        elif angle == 180 and p.dx > min_x:
            start_straight_length_section = p.dx - min_x
        else:
            start_straight_length_section = 0

        add_port(
            p,
            y_optical_top,
            routes,
            ports,
            start_straight_length=start_straight_length + start_straight_length_section,
        )
        y_optical_top += separation
        start_straight_length += separation

    start_straight_length_section = start_straight_length
    for p in backward_ports_thru_south:
        # Extend ports if necessary
        if angle == 0 and p.dx < max_x:
            start_straight_length_section = max_x - p.dx
        elif angle == 180 and p.dx > min_x:
            start_straight_length_section = p.dx - min_x
        else:
            start_straight_length_section = 0

        add_port(
            p,
            y_optical_bot,
            routes,
            ports,
            start_straight_length=start_straight_length + start_straight_length_section,
        )
        y_optical_bot -= separation
        start_straight_length += separation

    return routes, ports


def route_ports_to_y(
    component: Component,
    list_ports: list[Port],
    y: float | Literal["north", "south"] = "north",
    separation: float = 10.0,
    radius: float = 10.0,
    x0_left: float | None = None,
    x0_right: float | None = None,
    extension_length: float = 0.0,
    extend_left: float = 0.0,
    extend_right: float = 0.0,
    backward_port_side_split_index: int = 0,
    start_straight_length: float = 0.01,
    dx_start: float | None = None,
    dy_start: float | None = None,
    side: Literal["north", "south"] = "north",
    **routing_func_args: dict[Any, Any],
) -> tuple[list[ManhattanRoute], list[kf.Port]]:
    """Route ports to y.

    Args:
        component: component to route.
        list_ports: reasonably well behaved list of ports.
           ports facing north ports are norther than any other ports
           ports facing south ports are souther ...
           ports facing west ports are the wester ...
           ports facing east ports are the easter ...
        y: float or string.
               if float: y coordinate to which the ports will be routed
               if string: "north" -> route to north
               if string: "south" -> route to south
        separation: in um.
        radius: in um.
        x0_left: in um.
        x0_right: in um.
        extension_length: in um.
        extend_left: in um.
        extend_right: in um.
        backward_port_side_split_index: integer
               this integer represents and index in the list of backwards ports
                   (sorted from left to right)
               all ports with an index strictly larger are routed right
               all ports with an index lower or equal are routed left
        start_straight_length: in um.
        dx_start: override minimum starting x distance.
        dy_start: override minimum starting y distance.
        side: "north" or "south".
        routing_func_args: additional arguments to pass to the routing function.


    Returns:
        - a list of Routes
        - a list of the new optical ports

    First route the bottom-half of the back ports (back ports are the one facing opposite side of x)
    Then route the south ports
    then the front ports
    then the north ports
    """
    if y == "south" and extension_length > 0:
        extension_length = -extension_length

    da = 45
    north_ports = [
        p for p in list_ports if p.orientation > 90 - da and p.orientation < 90 + da
    ]
    south_ports = [
        p for p in list_ports if p.orientation > 270 - da and p.orientation < 270 + da
    ]
    east_ports = [
        p for p in list_ports if p.orientation < da or p.orientation > 360 - da
    ]
    west_ports = [
        p for p in list_ports if p.orientation < 180 + da and p.orientation > 180 - da
    ]

    epsilon = 1.0
    a = radius + max(radius, separation)
    bx = epsilon + max(radius, dx_start) if dx_start else a
    by = epsilon + max(radius, dy_start) if dy_start else a

    xs = [p.dx for p in list_ports]
    ys = [p.dy for p in list_ports]

    if x0_left is None:
        x0_left = min(xs) - bx
    x0_left -= extend_left

    if x0_right is None:
        x0_right = max(xs) + (max(radius, dx_start) if dx_start else a)
    x0_right += extend_right

    if y == "north":
        y = (
            max(
                p.dy + a * np.abs(np.cos(p.orientation * np.pi / 180))
                for p in list_ports
            )
            + by
        )
    elif y == "south":
        y = (
            min(
                p.dy - a * np.abs(np.cos(p.orientation * np.pi / 180))
                for p in list_ports
            )
            - by
        )
    elif isinstance(y, float | int):
        pass
    if y <= min(ys):
        sort_key_east = sort_key_south_to_north
        sort_key_west = sort_key_south_to_north
        forward_ports = south_ports
        backward_ports = north_ports

    elif y >= max(ys):
        sort_key_west = sort_key_north_to_south
        sort_key_east = sort_key_north_to_south
        forward_ports = north_ports
        backward_ports = south_ports
    else:
        raise ValueError("y should be either to the north or to the south of all ports")

    west_ports.sort(key=sort_key_west)
    east_ports.sort(key=sort_key_east)
    forward_ports.sort(key=sort_key_west_to_east)
    backward_ports.sort(key=sort_key_east_to_west)

    backward_ports.sort(key=sort_key_west_to_east)
    backward_ports_thru_west = backward_ports[:backward_port_side_split_index]
    backward_ports_thru_east = backward_ports[backward_port_side_split_index:]

    backward_ports_thru_west.sort(key=sort_key_west_to_east)
    backward_ports_thru_east.sort(key=sort_key_east_to_west)

    routes = []
    ports = []

    def add_port(
        p, x, l_elements, l_ports, start_straight_length=start_straight_length
    ):
        if side == "south":
            angle = 90

        elif side == "north":
            angle = 270

        new_port = p.copy()
        new_port.orientation = angle
        new_port.dcenter = (x, y + extension_length)

        if np.sum(np.abs((np.array(new_port.center) - p.center) ** 2)) < 1:
            l_ports += [flipped(new_port)]
            return

        try:
            l_elements += [
                route_single(
                    component,
                    p,
                    new_port,
                    start_straight_length=start_straight_length,
                    radius=radius,
                    **routing_func_args,
                )
            ]
            l_ports += [flipped(new_port)]

        except Exception as error:
            raise ValueError(
                f"Could not connect {p.name!r} to {new_port.name!r} {error}"
            ) from error

    x_optical_left = x0_left
    for p in west_ports:
        add_port(p, x_optical_left, routes, ports)
        x_optical_left -= separation

    for p in forward_ports:
        add_port(p, p.dx, routes, ports)

    x_optical_right = x0_right
    for p in east_ports:
        add_port(p, x_optical_right, routes, ports)
        x_optical_right += separation

    start_straight_length_section = start_straight_length
    for p in backward_ports_thru_east:
        add_port(
            p,
            x_optical_right,
            routes,
            ports,
            start_straight_length=start_straight_length_section,
        )
        x_optical_right += separation
        start_straight_length_section += separation

    start_straight_length_section = start_straight_length
    for p in backward_ports_thru_west:
        add_port(
            p,
            x_optical_left,
            routes,
            ports,
            start_straight_length=start_straight_length_section,
        )
        x_optical_left -= separation
        start_straight_length_section += separation

    return routes, ports


if __name__ == "__main__":
    c = Component("sample_route_sides")
    cross_section = "strip"
    dummy = gf.components.nxn(
        north=2, south=2, west=2, east=2, cross_section=cross_section
    )
    dummy_ref = c << dummy
    routes = route_ports_to_side(
        c,
        dummy_ref.ports,
        "south",
        cross_section=cross_section,
        y=-91,
        x=-100,
        # radius=5
    )
    # sides = ["north", "south", "east", "west"]
    # d = 100
    # positions = [(0, 0), (d, 0), (d, d), (0, d)]

    # for pos, side in zip(positions, sides):
    #     dummy_ref = c << dummy
    #     dummy_ref.dcenter = pos
    #     routes = route_ports_to_side(c, dummy_ref.ports, side, layer=(1, 0))

    c.show()
