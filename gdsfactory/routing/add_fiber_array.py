from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import select_ports_optical
from gdsfactory.routing.route_fiber_array import route_fiber_array
from gdsfactory.typings import (
    ComponentSpec,
    ComponentSpecOrList,
    CrossSectionSpec,
    PortsFactory,
)


def add_fiber_array(
    component: ComponentSpec = "straight",
    grating_coupler: ComponentSpecOrList = "grating_coupler_te",
    gc_port_name: str = "o1",
    select_ports: PortsFactory = select_ports_optical,
    cross_section: CrossSectionSpec = "strip",
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    mirror_grating_coupler: bool = False,
    **kwargs: Any,
) -> Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        gc_port_name: grating coupler input port name.
        select_ports: function to select ports.
        cross_section: cross_section function.
        mirror_grating_coupler: if True, mirrors the grating coupler.
        kwargs: additional arguments.

    Keyword Args:
        bend: bend spec.
        straight: straight spec.
        fanout_length: if None, automatic calculation of fanout length.
        max_y0_optical: in um.
        with_loopback: True, adds loopback structures.
        with_loopback_inside: True, adds loopback structures inside the component.
        straight_separation: from edge to edge.
        list_port_labels: None, adds TM labels to port indices in this list.
        nb_optical_ports_lines: number of grating coupler lines.
        force_manhattan: False
        excluded_ports: list of port names to exclude when adding gratings.
        grating_indices: list of grating coupler indices.
        routing_straight: function to route.
        routing_method: route_single.
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        input_port_indexes: to connect.
        pitch: in um.
        radius: optional radius of the bend. Defaults to the cross_section.
        radius_loopback: optional radius of the loopback bend. Defaults to the cross_section.
        start_straight_length: length of the start straight.
        end_straight_length: length of the end straight.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.crossing()
        cc = gf.routing.add_fiber_array(
            component=c,
            grating_coupler=gf.components.grating_coupler_elliptical_te,
            with_loopback=False
        )
        cc.plot()

    """
    component = gf.get_component(component)

    if isinstance(grating_coupler, list):
        gc = grating_coupler[0]
    else:
        gc = grating_coupler
    gc = gf.get_component(gc)
    if mirror_grating_coupler:
        gc = gf.functions.mirror(gc)

    gc_port_names = [port.name for port in gc.ports]
    if gc_port_name not in gc_port_names:
        raise ValueError(f"gc_port_name = {gc_port_name!r} not in {gc_port_names}")

    orientation = gc.ports[gc_port_name].orientation

    grating_coupler = (
        [gf.get_component(i) for i in grating_coupler]
        if isinstance(grating_coupler, list)
        else gf.get_component(grating_coupler)
    )

    if int(orientation) != 180:
        raise ValueError(
            "add_fiber_array requires a grating coupler port facing west "
            f"(orientation = 180). "
            f"Got orientation = {orientation} degrees for port {gc_port_name!r}"
        )

    if gc_port_name not in gc.ports:
        raise ValueError(f"gc_port_name={gc_port_name!r} not in {list(gc.ports)}")

    component_new = Component()

    optical_ports = select_ports(component.ports)
    if not optical_ports:
        raise ValueError(f"No optical ports found in {component.name!r}")

    ref = component_new.add_ref(component)
    route_fiber_array(
        component_new,
        ref,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        cross_section=cross_section,
        select_ports=select_ports,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        **kwargs,
    )

    component_new.copy_child_info(component)
    return component_new
