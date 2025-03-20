from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import kfactory as kf
import klayout.db as kdb

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.route_bundle import get_min_spacing, route_bundle
from gdsfactory.routing.route_single import route_single
from gdsfactory.routing.utils import direction_ports_from_list_ports
from gdsfactory.typings import (
    BoundingBoxes,
    ComponentSpec,
    ComponentSpecOrList,
    Coordinates,
    CrossSectionSpec,
    PortsFactory,
    Strs,
)


def route_fiber_array(
    component: Component,
    component_to_route: Component | ComponentReference,
    pitch: float = 127.0,
    grating_coupler: ComponentSpecOrList = "grating_coupler_te",
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    fanout_length: float | None = None,
    max_y0_optical: None = None,
    with_loopback: bool = True,
    with_loopback_inside: bool = True,
    straight_separation: float = 6.0,
    straight_to_grating_spacing: float = 5.0,
    nb_optical_ports_lines: int = 1,
    force_manhattan: bool = False,
    excluded_ports: list[str] | None = None,
    grating_indices: list[int] | None = None,
    gc_port_name: str = "o1",
    gc_port_name_fiber: str = "o2",
    gc_rotation: int = -90,
    component_name: str | None = None,
    x_grating_offset: float = 0,
    port_names: Strs | None = None,
    select_ports: PortsFactory = select_ports_optical,
    radius: float | None = None,
    radius_loopback: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_width_mismatch: bool = False,
    port_type: str = "optical",
    route_width: float | None = None,
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    auto_taper: bool = True,
    waypoints: Coordinates | None = None,
    steps: Sequence[Mapping[str, int | float]] | None = None,
    bboxes: BoundingBoxes | None = None,
    avoid_component_bbox: bool = True,
    **kwargs: Any,
) -> Component:
    """Returns new component with fiber array.

    Args:
        component: top level component.
        component_to_route: component to route.
        pitch: pitch between the array.
        grating_coupler: grating coupler instance, function or list of functions.
        bend: for bends.
        straight: straight.
        fanout_length: target distance between gratings and the southmost component port.
            If None, automatically calculated.
        max_y0_optical: Maximum y coordinate at which the intermediate optical ports can be set.
            Usually fine to leave at None.
        with_loopback: If True, add compact loopback alignment ports.
        with_loopback_inside: If True, the loopback is inside the component.
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
        gc_port_name_fiber: grating_coupler port name, where to route fibers.
        gc_rotation: grating_coupler rotation (deg).
        layer_label: for measurement labels.
        component_name: name of component.
        x_grating_offset: x offset.
        port_names: port names to route_to_fiber_array.
        select_ports: function to select ports for which to add grating couplers.
        radius: optional radius of the bend. Defaults to the cross_section.
        radius_loopback: optional radius of the loopback bend. Defaults to the cross_section.
        cross_section: cross_section.
        allow_width_mismatch: allow width mismatch.
        port_type: port type.
        route_width: width of the route. If None, defaults to cross_section.width.
        start_straight_length: length of the start straight.
        end_straight_length: length of the end straight.
        auto_taper: taper length for the IO.
        waypoints: waypoints for the route.
        steps: steps for the route.
        bboxes: list bounding boxes to avoid for routing.
        avoid_component_bbox: avoid component bbox for routing.
        kwargs: route_bundle settings.
    """
    x = gf.get_cross_section(cross_section)

    component_name = component_name or component_to_route.name
    excluded_ports = excluded_ports or []
    if port_names is None:
        to_route = list(select_ports(component_to_route.ports))
        port_names = [p.name for p in to_route if p.name is not None]
    else:
        to_route = [component_to_route.ports[lbl] for lbl in port_names]

    to_route = [p for p in to_route if p.name not in excluded_ports]

    ports_not_terminated = [
        port for port in component_to_route.ports if port.name not in port_names
    ]

    n = len(to_route)

    # optical_ports_labels = [p.name for p in ports]
    # print(optical_ports_labels)
    if n == 0:
        return component

    # grating_coupler can either be a component/function or a list of components/functions
    if isinstance(grating_coupler, list):
        grating_couplers = [gf.get_component(g) for g in grating_coupler]
        grating_coupler = grating_couplers[0]
    else:
        grating_coupler = gf.get_component(grating_coupler)
        grating_couplers = [grating_coupler] * n

    gc_port_names = [port.name for port in grating_coupler.ports]
    if gc_port_name not in gc_port_names:
        raise ValueError(f"{gc_port_name!r} not in {gc_port_names}")

    # Now:
    # - grating_coupler is a single grating coupler
    # - grating_couplers is a list of grating couplers
    # Define the route filter to apply to connection methods

    if radius:
        bend90 = gf.get_component(bend, cross_section=cross_section, radius=radius)
    else:
        bend90 = gf.get_component(bend, cross_section=cross_section)

    # `delta_gr_min` Used to avoid crossing between straights in special cases
    # This could happen when abs(x_port - x_grating) <= 2 * radius
    dy = bend90.ysize
    delta_gr_min = 2 * dy + 1

    # Get the center along x axis
    x_c = round(sum(p.x for p in to_route) / n, 1)

    # Sort the list of optical ports:
    direction_ports = direction_ports_from_list_ports(to_route)
    separation = straight_separation

    k = len(to_route)
    k = k + 1 if k % 2 else k

    # Set routing type if not specified
    pxs = [p.x for p in to_route]
    is_big_component = (
        (k > 2)
        or (max(pxs) - min(pxs) > pitch - delta_gr_min)
        or (component_to_route.xsize > pitch)
    )

    def has_p(side: str) -> bool:
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
        component_to_route.ymin
        - fanout_length
        - grating_coupler.ports[gc_port_name].x
        - dy
    )
    y0_optical += -k / 2 * separation

    if max_y0_optical is not None:
        y0_optical = round(min(max_y0_optical, y0_optical), 1)

    # - First connect half of the north ports going from middle of list
    # down to first elements
    # - then connect west ports (top to bottom)
    # - then connect south ports (left to right)
    # - then east ports (bottom to top)
    # - then second half of the north ports (right to left)
    ports: list[gf.Port] = []
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

    nb_ports_per_line = n // nb_optical_ports_lines
    y_gr_gap = (k / nb_optical_ports_lines + 1) * separation
    gr_coupler_y_sep = grating_coupler.ysize + y_gr_gap + dy
    offset = (nb_ports_per_line - 1) * pitch / 2 - x_grating_offset
    io_gratings_lines: list[
        list[Component | kf.DInstance]
    ] = []  # [[gr11, gr12, gr13...], [gr21, gr22, gr23...] ...]

    grating_coupler_port_names = [p.name for p in grating_coupler.ports]
    with_fiber_port = gc_port_name_fiber in grating_coupler_port_names

    if grating_indices is None:
        grating_indices = list(range(nb_ports_per_line))
    else:
        assert len(grating_indices) == nb_ports_per_line

    # add grating couplers
    io_gratings: list[Component | kf.DInstance] = []
    gc_ports: list[gf.Port] = []
    for j in range(nb_optical_ports_lines):
        for i, gc in zip(grating_indices, grating_couplers):
            gc_ref = component << gc
            gc_ref.rotate(gc_rotation)
            gc_ref.x = x_c - offset + i * pitch
            gc_ref.ymax = y0_optical - j * gr_coupler_y_sep
            io_gratings += [gc_ref]

        io_gratings_lines += [io_gratings[:]]
        if with_fiber_port:
            ports += [grating.ports[gc_port_name_fiber] for grating in io_gratings]

    if force_manhattan:
        # 1) find the min x_distance between each grating and component port.
        # 2) If abs(min distance) < 2* bend radius then offset io_gratings by -min_distance
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
    gc_ports_tmp: list[gf.Port] = []
    for io_gratings in io_gratings_lines:
        gc_ports_tmp += [gc.ports[gc_port_name] for gc in io_gratings]
    min_y = get_min_spacing(to_route, gc_ports_tmp, separation=separation, radius=dy)
    delta_y = abs(to_route[0].y - gc_ports_tmp[0].y)

    if min_y > delta_y:
        for io_gratings in io_gratings_lines:
            for gr in io_gratings:
                gr.y += delta_y - min_y

    # If we add align ports, we need enough space for the bends
    io_gratings = io_gratings_lines[0]
    gc_ports = [gc.ports[gc_port_name] for gc in io_gratings]
    # c.shapes(c.kcl.layer(1, 10)).insert(component_to_route.bbox())

    _bboxes: list[kdb.DBox] = [kdb.DBox(*bbox) for bbox in bboxes or []]

    if avoid_component_bbox:
        bbox = component_to_route.bbox()
        _bboxes.append(kdb.DBox(bbox.left, bbox.bottom, bbox.right, bbox.top))

    route_bundle(
        component,
        ports1=list(to_route),
        ports2=list(gc_ports),
        separation=separation,
        bend=bend90,
        straight=straight,
        cross_section=cross_section,
        port_type=port_type,
        sort_ports=True,
        allow_width_mismatch=allow_width_mismatch,
        route_width=route_width,
        bboxes=_bboxes,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        auto_taper=auto_taper,
        waypoints=waypoints,
        steps=steps,
        **kwargs,
    )
    if gc_port_name_fiber not in grating_coupler_port_names:
        _gc_port_name_fiber = gc_port_names[0]
        if _gc_port_name_fiber is not None:
            gc_port_name_fiber = _gc_port_name_fiber
        else:
            raise ValueError(
                f"{gc_port_name_fiber!r} not in {grating_coupler_port_names}"
            )

    fiber_ports = [gc.ports[gc_port_name_fiber] for gc in io_gratings]

    component.ports = kf.DPorts(kcl=component.kcl)

    for component_port, port in zip(port_names, fiber_ports):
        component.add_port(name=component_port, port=port)

    component.add_ports(ports_not_terminated)
    if with_loopback:
        ii = [grating_indices[0] - 1, grating_indices[-1] + 1]
        gca1 = component << grating_coupler
        gca2 = component << grating_coupler
        gca1.rotate(gc_rotation)
        gca2.rotate(gc_rotation)

        gca1.x = x_c - offset + ii[0] * pitch
        gca2.x = x_c - offset + ii[1] * pitch

        gc_loopback_dymin = io_gratings_lines[0][0].ymin
        gca1.ymin = gc_loopback_dymin
        gca2.ymin = gc_loopback_dymin

        port0 = gca1[gc_port_name]
        port1 = gca2[gc_port_name]
        radius = radius_loopback or radius or x.radius
        assert radius is not None
        radius_dbu = component.kcl.to_dbu(radius)
        d_loop = straight_to_grating_spacing + radius + gca1.ysize
        d_loop_dbu = component.kcl.to_dbu(d_loop)

        waypoints_loopback = kf.routing.optical.route_loopback(
            port0.to_itype(),
            port1.to_itype(),
            bend90_radius=radius_dbu,
            inside=with_loopback_inside,
            d_loop=d_loop_dbu,
        )
        waypoints_loopback_ = [
            p.to_dtype(component.kcl.dbu) for p in waypoints_loopback
        ]
        bend90 = gf.get_component(
            bend, cross_section=cross_section, radius=radius_loopback
        )

        route_single(
            component,
            port1=port0,
            port2=port1,
            waypoints=waypoints_loopback_,
            straight=straight,
            bend=bend90,
            cross_section=cross_section,
        )
        port0 = gca1[gc_port_name_fiber]
        port1 = gca2[gc_port_name_fiber]
        component.add_port(name="loopback1", port=port0)
        component.add_port(name="loopback2", port=port1)
    return component


if __name__ == "__main__":

    @gf.cell
    def mzi_with_bend(radius: float = 10, **kwargs: Any) -> Component:
        c = gf.Component()
        bend = c.add_ref(gf.components.bend_euler(radius=radius, **kwargs))
        mzi = c.add_ref(gf.components.mzi(**kwargs))
        bend.connect("o1", mzi.ports["o2"])
        c.add_port(name="o1", port=mzi.ports["o1"])
        c.add_port(name="o2", port=bend.ports["o2"])
        return c

    gc = gf.components.grating_coupler_elliptical_te(taper_length=10)

    # component = gf.components.nxn(north=10, south=10, east=10, west=10)
    # component = gf.components.straight()
    # component = gf.components.mmi2x2()
    # component = gf.components.straight_heater_metal()
    # component = gf.components.ring_single()
    # component = gf.components.ring_double()
    component = gf.components.crossing()
    # component = gf.components.mzi_phase_shifter()
    # component = gf.components.nxn(north=10, south=10, east=10, west=10)
    # component= gf.c.straight(width=2, length=50)

    c = gf.Component()
    ref = c << component
    routes = route_fiber_array(
        c,
        ref,
        # steps=[dict(dy=-50), dict(dx=3)],
        # grating_coupler=gc,
        # with_loopback=True,
        # radius=10,
        # pitch=50,
        # port_names=["o1", "o2"],
        # with_loopback=False,
        # fanout_length=-200,
        # force_manhattan=False,
        # auto_taper=False,
        excluded_ports=["o1"],
    )
    c.show()
    c.pprint_ports()
