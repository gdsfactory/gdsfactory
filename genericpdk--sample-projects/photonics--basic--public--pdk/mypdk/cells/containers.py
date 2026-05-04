"""This module contains cells that contain other cells."""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Strs,
)

gc = "grating_coupler_elliptical"
pack_doe = gf.c.pack_doe
pack_doe_grid = gf.c.pack_doe_grid


@gf.cell
def add_fiber_array(
    component: ComponentSpec = "straight",
    grating_coupler=gc,
    gc_port_name: str = "o1",
    component_name: str | None = None,
    cross_section: CrossSectionSpec = "strip",
    gc_rotation: float = -90,
    radius_loopback: float = 10,
    **kwargs,
) -> Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        gc_port_name: grating coupler input port name.
        component_name: optional for the label.
        cross_section: cross_section function.
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        radius_loopback: optional radius of the loopback bend. Defaults to the cross_section.
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
        connected_port_list_ids: names of ports only for type 0 optical routing.
        nb_optical_ports_lines: number of grating coupler lines.
        force_manhattan: False
        excluded_ports: list of port names to exclude when adding gratings.
        grating_indices: list of grating coupler indices.
        routing_straight: function to route.
        routing_method: route_single.
        optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check.
        input_port_indexes: to connect.
        pitch: in um.
        radius: optional radius of the bend. Defaults to the cross_section.
        route_backwards: route from component to grating coupler or vice-versa.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.crossing()
        cc = gf.routing.add_fiber_array(
            component=c,
            optical_routing_type=2,
            grating_coupler=gf.components.grating_coupler_elliptical_te,
            with_loopback=False
        )
        cc.plot()

    """
    return gf.routing.add_fiber_array(
        component=component,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        gc_rotation=gc_rotation,
        component_name=component_name,
        cross_section=cross_section,
        radius_loopback=radius_loopback,
        **kwargs,
    )


@gf.cell
def add_fiber_single(
    component: ComponentSpec = "straight",
    grating_coupler=gc,
    gc_port_name: str = "o1",
    component_name: str | None = None,
    cross_section: CrossSectionSpec = "strip",
    taper: ComponentSpec | None = None,
    input_port_names: list[str] | tuple[str, ...] | None = None,
    pitch: float = 70,
    with_loopback: bool = True,
    loopback_spacing: float = 100.0,
    **kwargs,
) -> Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        gc_port_name: grating coupler input port name.
        component_name: optional for the label.
        cross_section: cross_section function.
        taper: taper spec.
        input_port_names: list of input port names to connect to grating couplers.
        pitch: spacing between fibers.
        with_loopback: adds loopback structures.
        loopback_spacing: spacing between loopback and test structure.
        kwargs: additional arguments.

    Keyword Args:
        bend: bend spec.
        straight: straight spec.
        fanout_length: if None, automatic calculation of fanout length.
        max_y0_optical: in um.
        with_loopback: True, adds loopback structures.
        straight_separation: from edge to edge.
        list_port_labels: None, adds TM labels to port indices in this list.
        connected_port_list_ids: names of ports only for type 0 optical routing.
        nb_optical_ports_lines: number of grating coupler lines.
        force_manhattan: False
        excluded_ports: list of port names to exclude when adding gratings.
        grating_indices: list of grating coupler indices.
        routing_straight: function to route.
        routing_method: route_single.
        optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check.
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        input_port_indexes: to connect.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.crossing()
        cc = gf.routing.add_fiber_array(
            component=c,
            optical_routing_type=2,
            grating_coupler=gf.components.grating_coupler_elliptical_te,
            with_loopback=False
        )
        cc.plot()

    """
    return gf.routing.add_fiber_single(
        component=component,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        component_name=component_name,
        cross_section=cross_section,
        taper=taper,
        input_port_names=input_port_names,
        pitch=pitch,
        with_loopback=with_loopback,
        loopback_spacing=loopback_spacing,
        **kwargs,
    )


@gf.cell
def add_pads_top(
    component: ComponentSpec = "straight_metal",
    port_names: Strs | None = None,
    component_name: str | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
    pad_port_name: str = "e1",
    pad: ComponentSpec = "pad",
    bend: ComponentSpec = "wire_corner",
    straight_separation: float = 15.0,
    pad_pitch: float = 100.0,
    taper: ComponentSpec | None = None,
    port_type: str = "electrical",
    allow_width_mismatch: bool = True,
    fanout_length: float | None = 80,
    route_width: float | list[float] | None = 0,
    **kwargs,
) -> Component:
    """Returns new component with ports connected top pads.

    Args:
        component: component spec to connect to.
        port_names: list of port names to connect to pads.
        component_name: optional for the label.
        cross_section: cross_section function.
        pad_port_name: pad port name.
        pad: pad function.
        bend: bend function.
        straight_separation: from edge to edge.
        pad_pitch: spacing between pads.
        taper: taper function.
        port_type: port type.
        allow_width_mismatch: if True, allows width mismatch.
        fanout_length: length of the fanout.
        route_width: width of the route.
        kwargs: additional arguments.

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
    return gf.routing.add_pads_top(
        component=component,
        port_names=port_names,
        component_name=component_name,
        cross_section=cross_section,
        pad_port_name=pad_port_name,
        pad=pad,
        bend=bend,
        straight_separation=straight_separation,
        pad_pitch=pad_pitch,
        taper=taper,
        port_type=port_type,
        allow_width_mismatch=allow_width_mismatch,
        fanout_length=fanout_length,
        route_width=route_width,
        **kwargs,
    )
