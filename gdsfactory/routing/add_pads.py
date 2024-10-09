from __future__ import annotations

from collections.abc import Callable

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.pad import pad_rectangular
from gdsfactory.components.straight_heater_metal import straight_heater_metal
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing.route_fiber_array import route_fiber_array
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Strs,
)


def add_pads_bot(
    component: ComponentSpec = straight_heater_metal,
    select_ports: Callable = select_ports_electrical,
    port_names: Strs | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
    pad_port_name: str = "e1",
    pad: ComponentSpec = pad_rectangular,
    bend: ComponentSpec = "wire_corner",
    straight_separation: float = 15.0,
    pad_spacing: float | str = "pad_spacing",
    optical_routing_type: int | None = 1,
    port_type: str = "electrical",
    allow_width_mismatch: bool = True,
    fanout_length: float | None = 0,
    route_width: float | list[float] | None = 0,
    **kwargs,
) -> Component:
    """Returns new component with ports connected bottom pads.

    Args:
        component: component spec to connect to.
        select_ports: function to select_ports.
        port_names: optional port names. Overrides select_ports.
        cross_section: cross_section spec.
        get_input_labels_function: function to get input labels. None skips labels.
        layer_label: optional layer for grating coupler label.
        pad_port_name: pad input port name.
        pad_port_labels: pad list of labels.
        pad: spec for route terminations.
        bend: bend spec.
        straight_separation: from wire edge to edge. Defaults to xs.width+xs.gap
        pad_spacing: in um. Defaults to pad_spacing constant from the PDK.
        optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check.
        port_type: port type.
        allow_width_mismatch: True
        fanout_length: if None, automatic calculation of fanout length.
        route_width: width of the route. If None, defaults to cross_section.width.
        kwargs: additional arguments.

    Keyword Args:
        straight: straight spec.
        get_input_label_text_loopback_function: function to get input label test.
        get_input_label_text_function: for labels.
        max_y0_optical: in um.
        with_loopback: True, adds loopback structures.
        list_port_labels: None, adds TM labels to port indices in this list.
        connected_port_list_ids: names of ports only for type 0 optical routing.
        nb_optical_ports_lines: number of grating coupler lines.
        force_manhattan: False
        excluded_ports: list of port names to exclude when adding gratings.
        grating_indices: list of grating coupler indices.
        routing_straight: function to route.
        routing_method: route_single.
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        input_port_indexes: to connect.
        allow_width_mismatch: True

    .. plot::
        :include-source:

        import gdsfactory as gf
        c = gf.c.nxn(
            xsize=600,
            ysize=200,
            north=2,
            south=3,
            wg_width=10,
            layer="M3",
            port_type="electrical",
        )
        cc = gf.routing.add_pads_bot(component=c, port_names=("e1", "e4"), fanout_length=50)
        cc.plot()

    """
    component_new = Component()
    component = gf.get_component(component)

    pad_spacing = gf.get_constant(pad_spacing)
    cref = component_new << component
    ports = [cref[port_name] for port_name in port_names] if port_names else None
    ports = ports or select_ports(cref.ports)

    pad_component = gf.get_component(pad)
    if pad_port_name not in pad_component.ports:
        pad_ports = list(pad_component.ports)
        raise ValueError(
            f"pad_port_name = {pad_port_name!r} not in {pad_component.name!r} ports {pad_ports}"
        )

    pad_orientation = int(pad_component[pad_port_name].orientation)
    if pad_orientation != 180:
        raise ValueError(
            f"port.orientation={pad_orientation} for port {pad_port_name!r} needs to be 180 degrees."
        )

    if not ports:
        raise ValueError(
            f"select_ports or port_names did not match any ports in {list(component.ports)}"
        )

    route_fiber_array(
        component_new,
        component,
        grating_coupler=pad,
        gc_port_name=pad_port_name,
        cross_section=cross_section,
        select_ports=select_ports,
        with_loopback=False,
        bend=bend,
        straight_separation=straight_separation,
        port_names=port_names,
        fiber_spacing=pad_spacing,
        optical_routing_type=optical_routing_type,
        port_type=port_type,
        gc_port_name_fiber=pad_port_name,
        allow_width_mismatch=allow_width_mismatch,
        fanout_length=fanout_length,
        route_width=route_width,
        **kwargs,
    )
    component_new.add_ref(component)
    component_new.copy_child_info(component)
    return component_new


def add_pads_top(
    component: ComponentSpec = straight_heater_metal,
    select_ports: Callable = select_ports_electrical,
    port_names: Strs | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
    pad_port_name: str = "e1",
    pad: ComponentSpec = pad_rectangular,
    bend: ComponentSpec = "wire_corner",
    straight_separation: float = 15.0,
    pad_spacing: float | str = "pad_spacing",
    optical_routing_type: int | None = 1,
    port_type: str = "electrical",
    allow_width_mismatch: bool = True,
    fanout_length: float | None = 0,
    route_width: float | list[float] | None = 0,
    **kwargs,
) -> Component:
    """Returns new component with ports connected top pads.

    Args:
        component: component spec to connect to.
        select_ports: function to select_ports.
        port_names: optional port names. Overrides select_ports.
        cross_section: cross_section spec.
        get_input_labels_function: function to get input labels. None skips labels.
        layer_label: optional layer for grating coupler label.
        pad_port_name: pad input port name.
        pad_port_labels: pad list of labels.
        pad: spec for route terminations.
        bend: bend spec.
        straight_separation: from wire edge to edge. Defaults to xs.width+xs.gap
        pad_spacing: in um. Defaults to pad_spacing constant from the PDK.
        optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check.
        port_type: port type.
        allow_width_mismatch: True
        fanout_length: if None, automatic calculation of fanout length.
        route_width: width of the route. If None, defaults to cross_section.width.
        kwargs: additional arguments.

    Keyword Args:
        straight: straight spec.
        get_input_label_text_loopback_function: function to get input label test.
        get_input_label_text_function: for labels.
        max_y0_optical: in um.
        with_loopback: True, adds loopback structures.
        list_port_labels: None, adds TM labels to port indices in this list.
        connected_port_list_ids: names of ports only for type 0 optical routing.
        nb_optical_ports_lines: number of grating coupler lines.
        force_manhattan: False
        excluded_ports: list of port names to exclude when adding gratings.
        grating_indices: list of grating coupler indices.
        routing_straight: function to route.
        routing_method: route_single.
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        input_port_indexes: to connect.
        allow_width_mismatch: True

    .. plot::
        :include-source:

        import gdsfactory as gf
        c = gf.c.nxn(
            xsize=600,
            ysize=200,
            north=2,
            south=3,
            wg_width=10,
            layer="M3",
            port_type="electrical",
        )
        cc = gf.routing.add_pads_top(component=c, port_names=("e1", "e4"), fanout_length=50)
        cc.plot()

    """
    c = Component()
    _c = add_pads_bot(
        component=component,
        select_ports=select_ports,
        port_names=port_names,
        cross_section=cross_section,
        pad_port_name=pad_port_name,
        pad=pad,
        bend=bend,
        straight_separation=straight_separation,
        pad_spacing=pad_spacing,
        optical_routing_type=optical_routing_type,
        port_type=port_type,
        allow_width_mismatch=allow_width_mismatch,
        fanout_length=fanout_length,
        route_width=route_width,
        **kwargs,
    )
    ref = c << _c
    ref.mirror_y()
    c.add_ports(ref.ports)
    c.copy_child_info(_c)
    return c


if __name__ == "__main__":
    # c = gf.components.pad()
    c = gf.components.straight_heater_metal(length=100.0)
    # c = gf.components.straight(length=100.0)
    # c.pprint_ports()
    c = gf.routing.add_pads_top(component=c, port_names=("l_e1",))
    # c = gf.routing.add_pads_bot(component=c, port_names=("l_e4", "r_e4"), fanout_length=80)
    # c = gf.routing.add_fiber_array(c)
    c.show()
    # c.show()

    # cc = add_pads_top(component=c, port_names=("e1",))
    # cc = add_pads_top(component=c, port_names=("e1", "e2"), fanout_length=50)
    # c = gf.c.nxn(
    #     xsize=600,
    #     ysize=200,
    #     # north=2,
    #     # south=3,
    #     north=0,
    #     south=0,
    #     west=2,
    #     east=2,
    #     wg_width=10,
    #     layer="M3",
    #     port_type="electrical",
    # )
    # cc = add_pads_top(component=c)
    # cc.show()
