from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import phidl.device_layout as pd

from pp.component import Component, ComponentReference
from pp.config import conf
from pp.port import Port
from pp.routing.connect import connect_strip_way_points, get_waypoints_connect_strip
from pp.routing.utils import direction_ports_from_list_ports, flip


def route_south(
    component: Component,
    bend_radius: float = conf.tech.bend_radius,
    optical_routing_type: int = 1,
    excluded_ports: List[str] = None,
    waveguide_separation: float = 4.0,
    io_gratings_lines: Optional[List[List[ComponentReference]]] = None,
    route_filter: Callable = connect_strip_way_points,
    gc_port_name: str = "E0",
) -> Union[Tuple[List[Any], List[Port]], Tuple[List[ComponentReference], List[Port]]]:
    """
    Args:
        component: component to route
        bend_radius
        optical_routing_type: routing heuristic `1` or `2`
            `1` uses the component size info to estimate the box size.
            `2` only looks at the optical port positions to estimate the size
        excluded_ports=[]: list of port names to NOT route
        waveguide_separation
        io_gratings_lines: list of ports to which the ports produced by this
            function will be connected. Supplying this information helps
            avoiding waveguide collisions

        routing_method: routing method to connect the waveguides
        gc_port_name: grating port name

    Returns:
        list of elements, list of ports


    Works well if the component looks rougly like a rectangular box with
        north ports on the north of the box
        south ports on the south of the box
        east ports on the east of the box
        west ports on the west of the box
    """
    excluded_ports = excluded_ports or []
    assert optical_routing_type in [
        1,
        2,
    ], f"optical_routing_type = {optical_routing_type}, not supported "

    optical_ports = component.get_ports_list(port_type="optical")
    optical_ports = [p for p in optical_ports if p.name not in excluded_ports]
    csi = component.size_info
    elements = []

    # Handle empty list gracefully
    if not optical_ports:
        return [], []

    conn_params = {"bend_radius": bend_radius}

    route_filter_params = {
        "bend_radius": bend_radius,
        "wg_width": optical_ports[0].width,
    }

    def routing_method(p1, p2, **kwargs):
        way_points = get_waypoints_connect_strip(p1, p2, **kwargs)
        return route_filter(way_points, **route_filter_params)

    # Used to avoid crossing between waveguides in special cases
    # This could happen when abs(x_port - x_grating) <= 2 * bend_radius
    delta_gr_min = 2 * bend_radius + 1

    sep = waveguide_separation

    # Get lists of optical ports by orientation
    direction_ports = direction_ports_from_list_ports(optical_ports)

    north_ports = direction_ports["N"]
    north_start = north_ports[0 : len(north_ports) // 2]
    north_finish = north_ports[len(north_ports) // 2 :]

    west_ports = direction_ports["W"]
    west_ports.reverse()
    east_ports = direction_ports["E"]
    south_ports = direction_ports["S"]
    north_finish.reverse()  # Sort right to left
    north_start.reverse()  # Sort right to left
    ordered_ports = north_start + west_ports + south_ports + east_ports + north_finish

    def get_index_port_closest_to_x(x, list_ports):
        return np.array([abs(x - p.ports[gc_port_name].x) for p in list_ports]).argmin()

    def gen_port_from_port(x, y, p):
        new_p = pd.Port(name=p.name, midpoint=(x, y), orientation=90.0, width=p.width)

        return new_p

    R = bend_radius
    west_ports.reverse()

    y0 = min([p.y for p in ordered_ports]) - R - 0.5

    ports_to_route = []

    i = 0
    optical_xs_tmp = [p.x for p in ordered_ports]
    x_optical_min = min(optical_xs_tmp)
    x_optical_max = max(optical_xs_tmp)

    """
    ``x`` is the x-coord of the waypoint where the current component port is connected.
    x starts as close as possible to the component.
    For each new port, the distance is increased by the separation.
    The starting x depends on the heuristic chosen : ``1`` or ``2``
    """

    # Set starting ``x`` on the west side
    if optical_routing_type == 1:
        # use component size to know how far to route
        x = csi.west - R - 1
    elif optical_routing_type == 2:
        # use optical port to know how far to route
        x = x_optical_min - R - 1
    else:
        raise ValueError("Invalid optical routing type")

    # First route the ports facing west
    for p in west_ports:
        """
        In case we have to connect these ports to a line of grating,
        Ensure that the port is aligned with the grating port or
        has enough space for manhattan routing (at least two bend radius)
        """
        if io_gratings_lines:
            i_grating = get_index_port_closest_to_x(x, io_gratings_lines[-1])
            x_gr = io_gratings_lines[-1][i_grating].ports[gc_port_name].x
            if abs(x - x_gr) < delta_gr_min:
                if x > x_gr:
                    x = x_gr
                elif x < x_gr:
                    x = x_gr - delta_gr_min

        tmp_port = gen_port_from_port(x, y0, p)
        ports_to_route.append(tmp_port)
        elements.extend(routing_method(p, tmp_port, **conn_params)["references"])
        x -= sep

        i += 1
    start_straight = 0.5

    # First-half of north ports
    # This ensures that north ports are routed above the top west one
    north_start.reverse()  # We need them from left to right
    if len(north_start) > 0:
        y_max = max([p.y for p in west_ports + north_start])
        for p in north_start:
            tmp_port = gen_port_from_port(x, y0, p)

            elements.extend(
                routing_method(
                    p,
                    tmp_port,
                    start_straight=start_straight + y_max - p.y,
                    **conn_params,
                )["references"]
            )

            ports_to_route.append(tmp_port)
            x -= sep
            start_straight += sep

    # Set starting ``x`` on the east side
    if optical_routing_type == 1:
        #  use component size to know how far to route
        x = csi.east + R + 1
    elif optical_routing_type == 2:
        # use optical port to know how far to route
        x = x_optical_max + R + 1
    else:
        raise ValueError(
            f"Invalid optical routing type. Got {optical_routing_type}, only (1, 2 supported) "
        )
    i = 0

    # Route the east ports
    start_straight = 0.5
    for p in east_ports:
        """
        In case we have to connect these ports to a line of grating,
        Ensure that the port is aligned with the grating port or
        has enough space for manhattan routing (at least two bend radius)
        """
        if io_gratings_lines:
            i_grating = get_index_port_closest_to_x(x, io_gratings_lines[-1])
            x_gr = io_gratings_lines[-1][i_grating].ports[gc_port_name].x
            if abs(x - x_gr) < delta_gr_min:
                if x < x_gr:
                    x = x_gr
                elif x > x_gr:
                    x = x_gr + delta_gr_min

        tmp_port = gen_port_from_port(x, y0, p)

        elements.extend(
            routing_method(p, tmp_port, start_straight=start_straight, **conn_params)[
                "references"
            ]
        )

        ports_to_route.append(tmp_port)
        x += sep
        i += 1

    # Route the remaining north ports
    start_straight = 0.5
    if len(north_finish) > 0:
        y_max = max([p.y for p in east_ports + north_finish])
        for p in north_finish:
            tmp_port = gen_port_from_port(x, y0, p)
            ports_to_route.append(tmp_port)
            elements.extend(
                routing_method(
                    p,
                    tmp_port,
                    start_straight=start_straight + y_max - p.y,
                    **conn_params,
                )["references"]
            )
            x += sep
            start_straight += sep

    # Add south ports
    ports = [flip(p) for p in ports_to_route] + south_ports

    return elements, ports


if __name__ == "__main__":
    import pp

    c = pp.c.mmi2x2()
    elements, ports = route_south(c)
    for e in elements:
        if isinstance(e, list):
            print(len(e))
            print(e)
        # print(e)
        c.add(e)

    pp.show(c)
