from __future__ import annotations

from collections.abc import Callable

import kfactory as kf

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.route_bundle import get_min_spacing, route_bundle
from gdsfactory.routing.route_single import route_single
from gdsfactory.routing.route_south import route_south
from gdsfactory.routing.utils import direction_ports_from_list_ports
from gdsfactory.typings import (
    ComponentReference,
    ComponentSpec,
    ComponentSpecOrList,
    CrossSectionSpec,
    Strs,
)


def route_fiber_array(
    component: Component,
    component_to_route: Component | ComponentReference,
    fiber_spacing: float = 127.0,
    grating_coupler: ComponentSpecOrList = grating_coupler_te,
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    taper: ComponentSpec = taper_function,
    fanout_length: float | None = None,
    max_y0_optical: None = None,
    with_loopback: bool = True,
    straight_separation: float = 6.0,
    straight_to_grating_spacing: float = 5.0,
    nb_optical_ports_lines: int = 1,
    force_manhattan: bool = False,
    excluded_ports: list[str] | None = None,
    grating_indices: list[int] | None = None,
    gc_port_name: str = "o1",
    gc_rotation: int = -90,
    component_name: str | None = None,
    x_grating_offset: float = 0,
    port_names: Strs | None = None,
    select_ports: Callable = select_ports_optical,
    radius: float | None = None,
    cross_section: CrossSectionSpec = strip,
    optical_routing_type: int = 1,
    port_type: str = "optical",
) -> Component:
    """Returns new component with fiber array.

    Args:
        component: top level component.
        component_to_route: component to route.
        fiber_spacing: spacing between the optical fibers.
        grating_coupler: grating coupler instance, function or list of functions.
        bend: for bends.
        straight: straight.
        fanout_length: target distance between gratings and the southmost component port.
            If None, automatically calculated.
        max_y0_optical: Maximum y coordinate at which the intermediate optical ports can be set.
            Usually fine to leave at None.
        with_loopback: If True, add compact loopback alignment ports.
        straight_separation: min separation between routing straights.
        straight_to_grating_spacing: from align ports.
        nb_optical_ports_lines: number of lines with I/O grating couplers. One line by default.
            WARNING: Only works properly if:
            - nb_optical_ports_lines divides the total number of ports.
            - the components have an equal number of inputs and outputs.
        force_manhattan: sometimes port linker defaults to an S-bend due to lack of space to do manhattan.
            Force manhattan offsets all the ports to replace the s-bend by a straight link.
            This fails if multiple ports have the same issue.
        excluded_ports: ports excluded from routing.
        grating_indices: allows to fine skip some grating slots.
            e.g [0,1,4,5] will put two gratings separated by the pitch.
            Then there will be two empty grating slots, and after that an additional two gratings.
        gc_port_name: grating_coupler port name, where to route straights.
        gc_rotation: grating_coupler rotation (deg).
        layer_label: for measurement labels.
        component_name: name of component.
        x_grating_offset: x offset.
        port_names: port labels to route_to_fiber_array.
        select_ports: function to select ports for which to add grating couplers.
        radius: optional radius of the bend. Defaults to the cross_section.
        cross_section: cross_section.
    """
    if optical_routing_type not in [1, 2]:
        raise ValueError(f"optical_routing_type={optical_routing_type} must be 1 or 2")

    c = component
    component = component_to_route
    fiber_spacing = gf.get_constant(fiber_spacing)
    cross_section = x = gf.get_cross_section(cross_section)
    if radius:
        cross_section = x = cross_section.copy(radius=radius)

    component_name = component_name or component.name
    excluded_ports = excluded_ports or []
    if port_names is None:
        ports = list(select_ports(component.ports))
    else:
        ports = [component.ports[lbl] for lbl in port_names]

    ports = [p for p in ports if p.name not in excluded_ports]
    N = len(ports)

    # optical_ports_labels = [p.name for p in ports]
    # print(optical_ports_labels)
    if N == 0:
        return [], [], 0

    # grating_coupler can either be a component/function or a list of components/functions
    if isinstance(grating_coupler, list):
        grating_couplers = [gf.get_component(g) for g in grating_coupler]
        grating_coupler = grating_couplers[0]
    else:
        grating_coupler = gf.get_component(grating_coupler)
        grating_couplers = [grating_coupler] * N

    gc_port_names = [port.name for port in grating_coupler.ports]
    if gc_port_name not in gc_port_names:
        raise ValueError(f"{gc_port_name!r} not in {gc_port_names}")

    # Now:
    # - grating_coupler is a single grating coupler
    # - grating_couplers is a list of grating couplers
    # Define the route filter to apply to connection methods

    bend90 = gf.get_component(bend, cross_section=x)

    # `delta_gr_min` Used to avoid crossing between straights in special cases
    # This could happen when abs(x_port - x_grating) <= 2 * radius
    dy = bend90.d.ysize
    delta_gr_min = 2 * dy + 1

    # Get the center along x axis
    x_c = round(sum(p.d.x for p in ports) / N, 1)

    # Sort the list of optical ports:
    direction_ports = direction_ports_from_list_ports(ports)
    separation = straight_separation

    K = len(ports)
    K = K + 1 if K % 2 else K

    # Set routing type if not specified
    pxs = [p.d.x for p in ports]
    is_big_component = (
        (K > 2)
        or (max(pxs) - min(pxs) > fiber_spacing - delta_gr_min)
        or (component.d.xsize > fiber_spacing)
    )

    def has_p(side) -> bool:
        return len(direction_ports[side]) > 0

    list_ew_ports_on_sides = [has_p(side) for side in ["E", "W"]]
    list_ns_ports_on_sides = [has_p(side) for side in ["N", "S"]]

    has_ew_ports = any(list_ew_ports_on_sides)
    has_ns_ports = any(list_ns_ports_on_sides)

    is_one_sided_horizontal = False
    for side1, side2 in [("E", "W"), ("W", "E")]:
        if len(direction_ports[side1]) >= 2 and all(
            len(direction_ports[side]) == 0 for side in ["N", "S", side2]
        ):
            is_one_sided_horizontal = True

    # Compute fanout length if not specified
    if fanout_length is None:
        fanout_length = dy + 1.0
        # We need 3 bends in that case to connect the most bottom port to the grating couplers
        if has_ew_ports and is_big_component:
            # print('big')
            fanout_length = max(fanout_length, 3 * dy + 1.0)

        if has_ns_ports or is_one_sided_horizontal:
            # print('one sided')
            fanout_length = max(fanout_length, 2 * dy + 1.0)

        if has_ew_ports and not is_big_component:
            # print('east_west_ports')
            fanout_length = max(fanout_length, dy + 1.0)

    fanout_length += dy

    # use x for grating coupler since we rotate it
    y0_optical = (
        component.d.ymin - fanout_length - grating_coupler.ports[gc_port_name].d.x
    )
    y0_optical += -K / 2 * separation

    if max_y0_optical is not None:
        y0_optical = round(min(max_y0_optical, y0_optical), 1)

    # - First connect half of the north ports going from middle of list
    # down to first elements
    # - then connect west ports (top to bottom)
    # - then connect south ports (left to right)
    # - then east ports (bottom to top)
    # - then second half of the north ports (right to left)
    ports = []
    north_ports = direction_ports["N"]
    north_start = north_ports[: len(north_ports) // 2]
    north_finish = north_ports[len(north_ports) // 2 :]

    west_ports = direction_ports["W"]
    west_ports.reverse()
    # east_ports = direction_ports["E"]
    # south_ports = direction_ports["S"]
    north_finish.reverse()  # Sort right to left
    north_start.reverse()  # Sort right to left
    # ordered_ports = north_start + west_ports + south_ports + east_ports + north_finish

    nb_ports_per_line = N // nb_optical_ports_lines
    y_gr_gap = (K / nb_optical_ports_lines + 1) * separation
    gr_coupler_y_sep = grating_coupler.d.ysize + y_gr_gap + dy
    offset = (nb_ports_per_line - 1) * fiber_spacing / 2 - x_grating_offset
    io_gratings_lines = []  # [[gr11, gr12, gr13...], [gr21, gr22, gr23...] ...]

    if grating_indices is None:
        grating_indices = list(range(nb_ports_per_line))
    else:
        assert len(grating_indices) == nb_ports_per_line

    io_gratings = []
    gc_ports = []

    for j in range(nb_optical_ports_lines):
        for i, gc in zip(grating_indices, grating_couplers):
            gc_ref = c << gc
            gc_ref.d.rotate(gc_rotation)
            gc_ref.d.x = x_c - offset + i * fiber_spacing
            gc_ref.d.ymax = y0_optical - j * gr_coupler_y_sep
            io_gratings += [gc_ref]

        io_gratings_lines += [io_gratings[:]]
        ports += [grating.ports[gc_port_name] for grating in io_gratings]

    route_south(
        c,
        component,
        optical_routing_type=optical_routing_type,
        excluded_ports=excluded_ports,
        straight_separation=straight_separation,
        io_gratings_lines=io_gratings_lines,
        gc_port_name=gc_port_name,
        bend=bend90,
        straight=straight,
        taper=taper,
        select_ports=select_ports,
        port_names=port_names,
        cross_section=cross_section,
        port_type=port_type,
    )
    to_route = c.ports

    if force_manhattan:
        # 1) find the min x_distance between each grating and component port.
        # 2) If abs(min distance) < 2* bend radius then offset io_gratings by -min_distance
        min_dist = 2 * dy + 10.0
        min_dist_threshold = 2 * dy + 1.0
        for io_gratings in io_gratings_lines:
            for gr in io_gratings:
                for p in to_route:
                    dist = gr.d.x - p.d.x
                    if abs(dist) < abs(min_dist):
                        min_dist = dist
            if abs(min_dist) < min_dist_threshold:
                for gr in io_gratings:
                    gr.d.movex(-min_dist)

    # If the array of gratings is too close, adjust its location
    gc_ports_tmp = []
    for io_gratings in io_gratings_lines:
        gc_ports_tmp += [gc.ports[gc_port_name] for gc in io_gratings]
    min_y = get_min_spacing(to_route, gc_ports_tmp, separation=separation, radius=dy)
    delta_y = abs(to_route[0].d.y - gc_ports_tmp[0].d.y)

    if min_y > delta_y:
        for io_gratings in io_gratings_lines:
            for gr in io_gratings:
                gr.d.center = (gr.d.center.x, gr.d.center.y + delta_y - min_y)

    # If we add align ports, we need enough space for the bends
    if len(io_gratings_lines) == 1:
        io_gratings = io_gratings_lines[0]
        gc_ports = [gc.ports[gc_port_name] for gc in io_gratings]
        route_bundle(
            c,
            ports1=to_route,
            ports2=gc_ports,
            separation=separation,
            straight=straight,
            bend=bend90,
            cross_section=cross_section,
            port_type=port_type,
            sort_ports=True,
            enforce_port_ordering=False,
        )

    else:
        for io_gratings in io_gratings_lines:
            gc_ports = [gc.ports[gc_port_name] for gc in io_gratings]
            nb_gc_ports = len(io_gratings)
            nb_ports_to_route = len(to_route)
            n0 = nb_ports_to_route / 2
            dn = nb_gc_ports / 2
            route_bundle(
                c,
                ports1=to_route[n0 - dn : n0 + dn],
                ports2=gc_ports,
                separation=separation,
                bend=bend90,
                straight=straight,
                cross_section=cross_section,
                port_type=port_type,
                sort_ports=True,
                enforce_port_ordering=False,
            )
            del to_route[n0 - dn : n0 + dn]

    c.add_ports(gc_ports)

    if with_loopback:
        ii = [grating_indices[0] - 1, grating_indices[-1] + 1]
        gca1 = c << grating_coupler
        gca2 = c << grating_coupler
        gca1.d.rotate(gc_rotation)
        gca2.d.rotate(gc_rotation)

        gca1.d.x = x_c - offset + ii[0] * fiber_spacing
        gca2.d.x = x_c - offset + ii[1] * fiber_spacing

        gca1.d.ymax = round(y0_optical - j * gr_coupler_y_sep)
        gca2.d.ymax = round(y0_optical - j * gr_coupler_y_sep)

        port0 = gca1.ports[gc_port_name]
        port1 = gca2.ports[gc_port_name]
        radius = radius or x.radius
        radius_dbu = round(radius / c.kcl.dbu)

        waypoints = kf.routing.optical.route_loopback(
            port0,
            port1,
            bend90_radius=radius_dbu,
            d_loop=round(straight_to_grating_spacing / c.kcl.dbu)
            + radius_dbu
            + gca1.ysize,
        )

        route_single(
            c,
            port1=port0,
            port2=port1,
            waypoints=waypoints,
            straight=straight,
            bend=bend90,
            cross_section=cross_section,
        )
        c.add_port(name="loopback1", port=port0)
        c.add_port(name="loopback2", port=port1)

    return c


def demo() -> None:
    gcte = gf.components.grating_coupler_te
    gctm = gf.components.grating_coupler_tm

    c = gf.components.straight(length=500)
    c = gf.components.mmi2x2()

    route_fiber_array(
        component=c,
        grating_coupler=[gcte, gctm, gcte, gctm],
        with_loopback=True,
        # bend=gf.components.bend_euler,
        bend=gf.components.bend_circular,
        radius=20,
        # force_manhattan=True
    )
    c.show()


if __name__ == "__main__":
    c = gf.Component()

    gc = gf.components.grating_coupler_elliptical_te(taper_length=30)

    component = gf.components.nxn(north=10, south=10, east=10, west=10)
    # component = gf.components.straight()
    # component = gf.components.mmi2x2()
    # component = gf.components.straight_heater_metal()
    # component = gf.components.ring_single()
    # component = gf.components.ring_double()

    ref = c << component
    routes = route_fiber_array(
        c,
        ref,
        grating_coupler=gc,
        with_loopback=True,
        radius=10,
        # with_loopback=False,
        optical_routing_type=1,
        # optical_routing_type=2,
        # fanout_length=200,
        force_manhattan=True,
    )
    c.show()
