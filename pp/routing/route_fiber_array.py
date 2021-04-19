from typing import Any, Callable, List, Optional, Tuple, Union

from numpy import float64
from phidl.device_layout import Label

import pp
from pp.component import Component, ComponentReference
from pp.components.bend_euler import bend_euler
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_te
from pp.components.straight import straight
from pp.components.taper import taper
from pp.port import select_optical_ports
from pp.routing.get_bundle import get_min_spacing, link_ports
from pp.routing.get_input_labels import get_input_labels
from pp.routing.get_route import get_route_from_waypoints
from pp.routing.manhattan import generate_manhattan_waypoints, round_corners
from pp.routing.route_south import route_south
from pp.routing.utils import direction_ports_from_list_ports
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory


def route_fiber_array(
    component: Component,
    optical_io_spacing: Optional[float] = None,
    grating_coupler: ComponentFactory = grating_coupler_te,
    bend_factory: ComponentFactory = bend_euler,
    straight_factory: ComponentFactory = straight,
    taper_factory: ComponentFactory = taper,
    fanout_length: Optional[int] = None,
    max_y0_optical: None = None,
    with_align_ports: bool = True,
    straight_separation: float = 6.0,
    straight_to_grating_spacing: float = 5.0,
    optical_routing_type: Optional[int] = None,
    bend_radius: Optional[float] = None,
    connected_port_list_ids: None = None,
    nb_optical_ports_lines: int = 1,
    force_manhattan: bool = False,
    excluded_ports: List[Any] = None,
    grating_indices: None = None,
    route_filter: Callable = get_route_from_waypoints,
    gc_port_name: str = "W0",
    gc_rotation: int = -90,
    layer_label: Optional[Tuple[int, int]] = None,
    component_name: Optional[str] = None,
    x_grating_offset: int = 0,
    optical_port_labels: None = None,
    get_input_labels_function: Callable = get_input_labels,
    select_ports: Callable = select_optical_ports,
    tech: Tech = TECH_SILICON_C,
    auto_widen: bool = True,
) -> Tuple[
    List[Union[ComponentReference, Label]], List[List[ComponentReference]], float64
]:
    """Returns component I/O elements for adding grating couplers with a fiber array
    Many components are fine with the default.

    Args:
        component: The component to connect.
        optical_io_spacing: the wanted spacing between the optical I/O
        grating_coupler: grating coupler instance, function or list of functions
        bend_factory: bend_circular
        straight_factory: straight
        fanout_length: target distance between gratings and the southest component port.
            If None, automatically calculated.
        max_y0_optical: Maximum y coordinate at which the intermediate optical ports can be set.
            Usually fine to leave at None.
        with_align_ports: If True, add compact loopback alignment ports
        straight_separation: min separation between routing straights
        optical_routing_type: There are three options for optical routing
           * ``0`` is very basic but can be more compact.
            Can also be used in combination with ``connected_port_list_ids``
            or to route some components which otherwise fail with type ``1``.
           * ``1`` is the standard routing.
           * ``2`` uses the optical ports as a guideline for the component's physical size
            (instead of using the actual component size).
            Useful where the component is large due to metal tracks
           * ``None: leads to an automatic decision based on size and number
           of I/O of the component.
        bend_radius: optional bend_radius (defaults to tech.bend_radius)
        connected_port_list_ids: only for type 0 optical routing.
            Can specify which ports goes to which grating assuming the gratings are ordered from left to right.
            e.g ['N0', 'W1','W0','E0','E1', 'N1' ] or [4,1,7,3]
        nb_optical_ports_lines: number of lines with I/O grating couplers. One line by default.
            WARNING: Only works properly if:
            - nb_optical_ports_lines divides the total number of ports
            - the components have an equal number of inputs and outputs
        force_manhattan: sometimes port linker defaults to an S-bend due to lack of space to do manhattan.
            Force manhattan offsets all the ports to replace the s-bend by a straight link.
            This fails if multiple ports have the same issue
        excluded_ports: ports excluded from routing
        grating_indices: allows to fine skip some grating slots
            e.g [0,1,4,5] will put two gratings separated by the pitch.
            Then there will be two empty grating slots, and after that an additional two gratings.
        route_filter: straight and bend factories
        gc_port_name: grating_coupler port name, where to route straights
        gc_rotation: grating_coupler rotation (deg)
        layer_label: for TM labels
        component_name: name of component
        x_grating_offset: x offset
        optical_port_labels: port labels that need connection
        get_input_labels_function: functions to add labels
        select_ports: function to select ports

    Returns:
        elements, io_grating_lines, y0_optical
    """
    layer_label = layer_label or tech.layer_label
    bend_radius = bend_radius or tech.bend_radius
    optical_io_spacing = optical_io_spacing or tech.fiber_array_spacing

    component_name = component_name or component.name
    excluded_ports = excluded_ports or []
    if optical_port_labels is None:
        # for pn, p in component.ports.items():
        #     print(p.name, p.port_type, p.layer)
        # optical_ports = component.get_ports_list(port_type='optical')
        optical_ports = list(select_ports(component.ports).values())
        # print(optical_ports)
    else:
        optical_ports = [component.ports[lbl] for lbl in optical_port_labels]

    optical_ports = [p for p in optical_ports if p.name not in excluded_ports]
    N = len(optical_ports)
    if N == 0:
        return [], [], 0

    elements = []

    # grating_coupler can either be a component/function
    # or a list of components/functions

    if isinstance(grating_coupler, list):
        grating_couplers = [pp.call_if_func(g) for g in grating_coupler]
        grating_coupler = grating_couplers[0]
    else:
        grating_coupler = pp.call_if_func(grating_coupler)
        grating_couplers = [grating_coupler] * N

    assert (
        gc_port_name in grating_coupler.ports
    ), f"{gc_port_name} not in {list(grating_coupler.ports.keys())}"

    # Now:
    # - grating_coupler is a single grating coupler
    # - grating_couplers is a list of grating couplers
    # Define the route filter to apply to connection methods

    bend90 = (
        bend_factory(radius=bend_radius) if callable(bend_factory) else bend_factory
    )

    dy = abs(bend90.dy)

    # `delta_gr_min` Used to avoid crossing between straights in special cases
    # This could happen when abs(x_port - x_grating) <= 2 * bend_radius

    dy = bend90.dy
    delta_gr_min = 2 * dy + 1

    io_sep = optical_io_spacing
    offset = (N - 1) * io_sep / 2.0

    # Get the center along x axis
    x_c = round(sum([p.x for p in optical_ports]) / N, 1)
    y_min = component.ymin  # min([p.y for p in optical_ports])

    # Sort the list of optical ports:
    direction_ports = direction_ports_from_list_ports(optical_ports)
    sep = straight_separation

    K = len(optical_ports)
    K = K + 1 if K % 2 else K

    # Set routing type if not specified
    pxs = [p.x for p in optical_ports]
    is_big_component = (
        (K > 2)
        or (max(pxs) - min(pxs) > io_sep - delta_gr_min)
        or (component.xsize > io_sep)
    )
    if optical_routing_type is None:
        if not is_big_component:
            optical_routing_type = 0
        else:
            optical_routing_type = 1

    # choose the default length if the default fanout distance is not set
    def has_p(side):
        return len(direction_ports[side]) > 0

    list_ew_ports_on_sides = [has_p(side) for side in ["E", "W"]]
    list_ns_ports_on_sides = [has_p(side) for side in ["N", "S"]]

    has_ew_ports = any(list_ew_ports_on_sides)
    has_ns_ports = any(list_ns_ports_on_sides)

    is_one_sided_horizontal = False
    for side1, side2 in [("E", "W"), ("W", "E")]:
        if len(direction_ports[side1]) >= 2:
            if all([len(direction_ports[side]) == 0 for side in ["N", "S", side2]]):
                is_one_sided_horizontal = True

    # Compute fanout length if not specified
    if fanout_length is None:
        fanout_length = dy + 1.0
        # We need 3 bends in that case to connect the most bottom port to the
        # grating couplers
        if has_ew_ports and is_big_component:
            # print('big')
            fanout_length = max(fanout_length, 3 * dy + 1.0)

        if has_ns_ports or is_one_sided_horizontal:
            # print('one sided')
            fanout_length = max(fanout_length, 2 * dy + 1.0)

        if has_ew_ports and not is_big_component:
            # print('ew_ports')
            fanout_length = max(fanout_length, dy + 1.0)

    fanout_length += dy

    # use x for grating coupler since we rotate it
    y0_optical = y_min - fanout_length - grating_coupler.ports[gc_port_name].x
    y0_optical += -K / 2 * sep
    y0_optical = round(y0_optical, 1)

    if max_y0_optical is not None:
        y0_optical = round(min(max_y0_optical, y0_optical), 1)
    """
     - First connect half of the north ports going from middle of list
    down to first elements
     - then connect west ports (top to bottom)
     - then connect south ports (left to right)
     - then east ports (bottom to top)
     - then second half of the north ports (right to left)

    """
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

    nb_ports_per_line = N // nb_optical_ports_lines
    grating_coupler_si = grating_coupler.size_info
    y_gr_gap = (K / (nb_optical_ports_lines) + 1) * sep
    gr_coupler_y_sep = grating_coupler_si.height + y_gr_gap + dy

    offset = (nb_ports_per_line - 1) * io_sep / 2 - x_grating_offset
    io_gratings_lines = []  # [[gr11, gr12, gr13...], [gr21, gr22, gr23...] ...]

    if grating_indices is None:
        grating_indices = list(range(nb_ports_per_line))
    else:
        assert len(grating_indices) == nb_ports_per_line

    for j in range(nb_optical_ports_lines):
        io_gratings = [
            gc.ref(
                position=(x_c - offset + i * io_sep, y0_optical - j * gr_coupler_y_sep),
                rotation=gc_rotation,
                port_id=gc_port_name,
            )
            for i, gc in zip(grating_indices, grating_couplers)
        ]

        io_gratings_lines += [io_gratings[:]]

    if optical_routing_type == 0:
        """
        Basic optical routing, typically fine for small components
        No heuristic to avoid collisions between connectors.

        If specified ports to connect in a specific order
        (i.e if connected_port_list_ids is not None and not empty)
        then grab these ports
        """
        if connected_port_list_ids:
            ordered_ports = [component.ports[i] for i in connected_port_list_ids]

        for io_gratings in io_gratings_lines:
            for i in range(N):
                p0 = io_gratings[i].ports[gc_port_name]
                p1 = ordered_ports[i]
                waypoints = generate_manhattan_waypoints(
                    input_port=p0, output_port=p1, bend_factory=bend90
                )
                route = route_filter(
                    waypoints=waypoints,
                    bend_factory=bend90,
                    auto_widen=auto_widen,
                )
                elements.extend(route["references"])

    elif optical_routing_type in [1, 2]:
        route = route_south(
            component=component,
            bend_radius=bend_radius,
            optical_routing_type=optical_routing_type,
            excluded_ports=excluded_ports,
            straight_separation=straight_separation,
            io_gratings_lines=io_gratings_lines,
            gc_port_name=gc_port_name,
            bend_factory=bend_factory,
            straight_factory=straight_factory,
            taper_factory=taper_factory,
            auto_widen=auto_widen,
        )
        elems = route["references"]
        to_route = route["ports"]
        elements.extend(elems)

        if force_manhattan:
            """
            1) find the min x_distance between each grating port and
            each component port.
            2) If abs(min distance) < 2* bend radius
                then offset io_gratings by -min_distance
            """
            min_dist = 2 * dy + 10.0
            min_dist_threshold = 2 * dy + 1.0
            for io_gratings in io_gratings_lines:
                for gr in io_gratings:
                    for p in to_route:
                        dist = gr.x - p.x
                        if abs(dist) < abs(min_dist):
                            min_dist = dist
                if abs(min_dist) < min_dist_threshold:
                    for gr in io_gratings:
                        gr.movex(-min_dist)

        # If the array of gratings is too close, adjust its location
        gc_ports_tmp = []
        for io_gratings in io_gratings_lines:
            gc_ports_tmp += [gc.ports[gc_port_name] for gc in io_gratings]
        min_y = get_min_spacing(to_route, gc_ports_tmp, sep=sep, radius=dy)
        delta_y = abs(to_route[0].y - gc_ports_tmp[0].y)

        if min_y > delta_y:
            for io_gratings in io_gratings_lines:
                for gr in io_gratings:
                    gr.translate(0, delta_y - min_y)

        # If we add align ports, we need enough space for the bends
        end_straight_offset = straight_separation + 5 if with_align_ports else 0.01
        if len(io_gratings_lines) == 1:
            io_gratings = io_gratings_lines[0]
            gc_ports = [gc.ports[gc_port_name] for gc in io_gratings]
            routes = link_ports(
                start_ports=to_route,
                end_ports=gc_ports,
                separation=sep,
                end_straight_offset=end_straight_offset,
                route_filter=route_filter,
                bend_factory=bend90,
                bend_radius=bend_radius,
                auto_widen=auto_widen,
            )
            elements.extend([route["references"] for route in routes])

        else:
            for io_gratings in io_gratings_lines:
                gc_ports = [gc.ports[gc_port_name] for gc in io_gratings]
                nb_gc_ports = len(io_gratings)
                nb_ports_to_route = len(to_route)
                n0 = nb_ports_to_route / 2
                dn = nb_gc_ports / 2
                routes = link_ports(
                    start_ports=to_route[n0 - dn : n0 + dn],
                    end_ports=gc_ports,
                    separation=sep,
                    end_straight_offset=end_straight_offset,
                    bend_factory=bend90,
                    route_filter=route_filter,
                    bend_radius=bend_radius,
                    auto_widen=auto_widen,
                )
                elements.extend([route["references"] for route in routes])
                del to_route[n0 - dn : n0 + dn]

    if with_align_ports:
        gca1, gca2 = [
            grating_coupler.ref(
                position=(
                    x_c - offset + ii * io_sep,
                    io_gratings_lines[-1][0].ports[gc_port_name].y,
                ),
                rotation=gc_rotation,
                port_id=gc_port_name,
            )
            for ii in [grating_indices[0] - 1, grating_indices[-1] + 1]
        ]

        p0 = gca1.ports[gc_port_name].position
        p1 = gca2.ports[gc_port_name].position

        dy = bend90.dy
        dx = max(2 * dy, io_sep / 2)

        gc_east = max([gci.size_info.east for gci in grating_couplers])
        y_bot_align_route = gc_east + straight_to_grating_spacing

        points = [
            p0,
            p0 + (0, dy),
            p0 + (dx, dy),
            p0 + (dx, -y_bot_align_route),
            p1 + (-dx, -y_bot_align_route),
            p1 + (-dx, dy),
            p1 + (0, dy),
            p1,
        ]
        elements.extend([gca1, gca2])

        route = round_corners(
            points=points, straight_factory=straight_factory, bend_factory=bend90
        )
        elements.extend(route["references"])

    elements.extend(
        get_input_labels_function(
            io_gratings, ordered_ports, component_name, layer_label, gc_port_name
        )
    )

    return elements, io_gratings_lines, y0_optical


def demo():
    gcte = pp.components.grating_coupler_te
    gctm = pp.components.grating_coupler_tm

    c = pp.components.straight(length=500)
    c = pp.components.mmi2x2()
    c = pp.components.mzi2x2()

    elements, gc, _ = route_fiber_array(
        component=c,
        grating_coupler=[gcte, gctm, gcte, gctm],
        with_align_ports=True,
        optical_routing_type=2,
        # bend_factory=pp.components.bend_euler,
        bend_factory=pp.components.bend_circular,
        bend_radius=20,
        # force_manhattan=True
    )
    for e in elements:
        # if isinstance(e, list):
        # print(len(e))
        # print(e)
        c.add(e)
    for e in gc:
        c.add(e)
    c.show()


if __name__ == "__main__":
    from pp.pdk import PDK_NITRIDE_C

    p = PDK_NITRIDE_C

    c = p.ring_single()
    c = p.add_fiber_array(c, optical_routing_type=1, auto_widen=False)
    c.show()
