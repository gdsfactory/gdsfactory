from typing import Any

import gdsfactory as gf
from gdsfactory._deprecation import deprecate
from gdsfactory.typings import AngleInDegrees, ComponentSpec, CrossSectionSpec


def add_fiber_array_optical_south_electrical_north(
    component: ComponentSpec,
    pad: ComponentSpec,
    grating_coupler: ComponentSpec,
    cross_section_metal: CrossSectionSpec,
    with_loopback: bool = True,
    pad_pitch: float = 100.0,
    pitch: float = 127.0,
    pad_gc_spacing: float = 250.0,
    electrical_port_names: list[str] | None = None,
    electrical_port_orientation: AngleInDegrees | None = 90,
    npads: int | None = None,
    port_types_grating_couplers: list[str] | None = None,
    pad_spacing: float | None = None,
    **kwargs: Any,
) -> gf.Component:
    """Returns a fiber array with Optical gratings on South and Electrical pads on North.

    This a test configuration for DC pads.

    Args:
        component: component spec to add fiber and pads.
        pad: pad spec.
        grating_coupler: grating coupler function.
        cross_section_metal: metal cross section.
        with_loopback: whether to add a loopback port.
        pad_pitch: spacing between pads.
        pitch: spacing between grating couplers.
        pad_gc_spacing: spacing between pads and grating couplers.
        electrical_port_names: list of electrical port names. Defaults to all.
        electrical_port_orientation: orientation of electrical ports. Defaults to 90.
        npads: number of pads. Defaults to one per electrical_port_names.
        port_types_grating_couplers: port types for grating couplers. Defaults to vertical TE, TM, and dual.
        pad_spacing: (deprecated).
        kwargs: additional arguments.

    Keyword Args:
        layer_label: layer for settings label.
        measurement: measurement name.
        measurement_settings: measurement settings.
        analysis: analysis name.
        doe: Design of Experiment.
        anchor: anchor point for the label. Defaults to south-west "sw". \
            Valid options are: "n", "s", "e", "w", "ne", "nw", "se", "sw", "c".
        gc_port_name: grating coupler input port name.
        gc_port_labels: grating coupler list of labels.
        component_name: optional for the label.
        select_ports: function to select ports.
        cross_section: cross_section function.
        get_input_labels_function: function to get input labels. None skips labels.
        layer_label: optional layer for grating coupler label.
        bend: bend spec.
        straight: straight spec.
        taper: taper spec.
        get_input_label_text_loopback_function: function to get input label test.
        get_input_label_text_function: for labels.
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
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        input_port_indexes: to connect.

    """
    if pad_spacing is not None:
        deprecate("pad_spacing", "pad_pitch")
        pad_pitch = pad_spacing

    c = gf.Component()
    component = gf.get_component(component)
    r = c << gf.routing.add_fiber_array(
        component=component,
        grating_coupler=grating_coupler,
        with_loopback=with_loopback,
        pitch=pitch,
        **kwargs,
    )
    port_types_grating_couplers = (
        port_types_grating_couplers or gf.CONF.port_types_grating_couplers
    )
    optical_ports = [
        port for port in r.ports if port.port_type in port_types_grating_couplers
    ]
    c.add_ports(optical_ports)

    electrical_ports = r.ports.filter(
        port_type="electrical", orientation=electrical_port_orientation
    )
    electrical_port_names_list = electrical_port_names or [
        p.name for p in electrical_ports if p.name is not None
    ]

    npads = npads or len(electrical_port_names_list)
    pads = c << gf.components.array(
        component=pad,
        columns=npads,
        column_pitch=pad_pitch,
    )
    pads.dx = r.dx
    pads.dymin = r.dymin + pad_gc_spacing

    electrical_ports = [r[por_name] for por_name in electrical_port_names_list]
    nroutes = min(len(electrical_ports), npads)

    ports1 = electrical_ports[:nroutes]
    ports2 = list(pads.ports.filter(orientation=270))[:nroutes]

    gf.routing.route_bundle_electrical(
        c,
        ports1=ports1,
        ports2=ports2,
        cross_section=cross_section_metal,
        sort_ports=True,
    )

    c.add_ports(ports2)
    return c


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory import cross_section as xs

    c = gf.c.add_fiber_array_optical_south_electrical_north(
        component=gf.c.straight_heater_metal,
        pad=gf.c.pad,  # type: ignore
        grating_coupler=gf.c.grating_coupler_te,  # type: ignore
        cross_section_metal=xs.metal_routing,
        pad_pitch=100,
    )
    c.pprint_ports()
    c.show()
