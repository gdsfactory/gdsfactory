import gdsfactory as gf
from gdsfactory.components.grating_coupler_elliptical import (
    grating_coupler_elliptical_te,
)
from gdsfactory.components.mzi import mzi_phase_shifter
from gdsfactory.components.pad import pad_small
from gdsfactory.typings import Callable, ComponentSpec, CrossSectionSpec


@gf.cell_with_child
def add_fiber_array_optical_south_electrical_north(
    component: ComponentSpec = mzi_phase_shifter,
    pad: ComponentSpec = pad_small,
    with_loopback: bool = True,
    pad_spacing: float = 100.0,
    fiber_spacing: float = 127.0,
    pad_gc_spacing: float = 250.0,
    electrical_port_names: list[str] | None = None,
    electrical_port_orientation: float | None = 90,
    npads: int | None = None,
    pad_assigments: tuple[tuple[str, str], ...] | None = None,
    grating_coupler: ComponentSpec = grating_coupler_elliptical_te,
    xs_metal: CrossSectionSpec = "xs_metal_routing",
    post_process: Callable | None = None,
    **kwargs,
) -> gf.Component:
    """Returns a fiber array with Optical gratings on South and Electrical pads on North.

    This a test configuration for DC pads.

    Args:
        component: component spec to add fiber and pads.
        pad: pad spec.
        with_loopback: whether to add a loopback port.
        pad_spacing: spacing between pads.
        fiber_spacing: spacing between grating couplers.
        pad_gc_spacing: spacing between pads and grating couplers.
        electrical_port_names: list of electrical port names. Defaults to all.
        electrical_port_orientation: orientation of electrical ports. Defaults to 90.
        npads: number of pads. Defaults to one per electrical_port_names.
        pad_assigments: if not None, routes according to (component_port_name, pad_port_name).
        grating_coupler: grating coupler function.
        xs_metal: metal cross section.
        post_process: function to run after the cutback is created.

    Keyword Args:
        gc_port_name: grating coupler input port name.
        gc_port_labels: grating coupler list of labels.
        io_rotation: fiber coupler rotation in degrees. Defaults to None.
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
        routing_method: get_route.
        optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check.
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        input_port_indexes: to connect.

    """
    c = gf.Component()
    component = gf.get_component(component)
    r = c << gf.routing.add_fiber_array(
        component=component,
        grating_coupler=grating_coupler,
        with_loopback=with_loopback,
        fiber_spacing=fiber_spacing,
        layer_label=None,
        **kwargs,
    )
    optical_ports = r.get_ports_list(port_type="optical")
    c.add_ports(optical_ports)

    electrical_ports = r.get_ports_list(
        port_type="electrical", orientation=electrical_port_orientation
    )

    if not electrical_ports:
        raise ValueError(
            f"No electrical ports found with orientation {electrical_port_orientation}. "
            f"{r.pprint_ports()}"
        )

    electrical_port_names = electrical_port_names or [p.name for p in electrical_ports]

    npads = npads or len(electrical_port_names)
    pads = c << gf.components.array(
        component=pad,
        columns=npads,
        spacing=(pad_spacing, 0),
    )
    pads.x = r.x
    pads.ymin = r.ymin + pad_gc_spacing

    electrical_ports = [r[por_name] for por_name in electrical_port_names]
    nroutes = min(len(electrical_ports), npads)

    if electrical_port_orientation and int(electrical_port_orientation) != 90:
        routes, electrical_ports = gf.routing.route_ports_to_side(
            ports=electrical_ports,
            side="north",
            cross_section=xs_metal,
            separation=pad_spacing,
        )
        for route in routes:
            c.add(route.references)

    ports1 = electrical_ports[:nroutes]
    pad_ports = pads.get_ports_list(orientation=270)

    if pad_assigments is None:
        ports2 = pads.get_ports_list(orientation=270)[:nroutes]
        routes = gf.routing.get_bundle_electrical(
            ports1=ports1,
            ports2=ports2,
            cross_section=xs_metal,
            enforce_port_ordering=False,
        )
    else:
        ports2 = []
        ports2_dict = pads.get_ports_dict()
        routes = []
        for port1_name, port2_name in pad_assigments:
            port1 = r[port1_name]
            port2 = ports2_dict[port2_name]
            route = gf.routing.get_route_electrical(input_port=port1, output_port=port2)
            routes.append(route)
            ports2.append(port2)

    for route in routes:
        c.add(route.references)

    c.add_ports(pad_ports)
    c.copy_child_info(component)

    if post_process:
        post_process(c)
    return c


if __name__ == "__main__":
    # c = add_fiber_array_optical_south_electrical_north(
    #     info=dict(wavelength_min=1550), name="my_mzi"
    # )

    # d = json.loads(c.labels[0].text)
    # print(d)
    # import gdsfactory as gf
    # from functools import partial

    # component = partial(mzi_phase_shifter, length_y=1)
    # c = add_fiber_array_optical_south_electrical_north(
    #     component=component,
    #     electrical_port_names=["top_l_e2", "top_r_e2"],
    #     npads=5,
    # )
    # component = partial(gf.c.ring_single_heater, length_x=10)
    # c = add_fiber_array_optical_south_electrical_north(
    #     component=component,
    #     electrical_port_names=["l_e2", "r_e2"],
    #     npads=5,
    # )
    # print(c.name)
    c = add_fiber_array_optical_south_electrical_north()
    c.show(show_ports=False)
