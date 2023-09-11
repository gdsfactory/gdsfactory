import gdsfactory as gf
from gdsfactory.components.grating_coupler_elliptical import (
    grating_coupler_elliptical_te,
)
from gdsfactory.components.mzi_phase_shifter import mzi_phase_shifter
from gdsfactory.components.pad import pad_small
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def add_fiber_array_optical_south_electrical_north(
    dut: ComponentSpec = mzi_phase_shifter,
    pad: ComponentSpec = pad_small,
    with_loopback: bool = True,
    pad_spacing: float = 100.0,
    fiber_spacing: float = 127.0,
    pad_gc_spacing: float = 250.0,
    electrical_port_names: list[str] | None = None,
    npads: int | None = None,
    grating_coupler: ComponentSpec = grating_coupler_elliptical_te,
    xs_metal: CrossSectionSpec = "metal_routing",
    **kwargs,
) -> gf.Component:
    """Returns a fiber array with Optical gratings on South and Electrical pads on North.

    Args:
        dut: device under test.
        pad: pad spec.
        with_loopback: whether to add a loopback port.
        pad_spacing: spacing between pads.
        fiber_spacing: spacing between grating couplers.
        pad_gc_spacing: spacing between pads and grating couplers.
        electrical_port_names: list of electrical port names. Defaults to all electrical ports.
        npads: number of pads. Defaults to one per electrical_port_names.
        grating_coupler: grating coupler function.
        xs_metal: metal cross section.

    Keyword Args:
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
        routing_method: get_route.
        optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check.
        gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
        input_port_indexes: to connect.

    """
    c = gf.Component()
    r = c << gf.routing.add_fiber_array(
        component=dut,
        grating_coupler=grating_coupler,
        with_loopback=with_loopback,
        fiber_spacing=fiber_spacing,
        **kwargs,
    )

    c.add_ports(r.ports)

    electrical_port_names = electrical_port_names or r.get_ports_list(
        port_type="electrical"
    )
    pads = c << gf.components.array(
        component=pad,
        columns=npads or len(electrical_port_names),
        spacing=(pad_spacing, 0),
    )
    pads.x = r.x
    pads.ymin = r.ymin + pad_gc_spacing

    electrical_ports = [r[por_name] for por_name in electrical_port_names]
    routes = gf.routing.get_bundle_electrical(
        ports1=electrical_ports,
        ports2=pads.get_ports_list(orientation=270),
        cross_section=xs_metal,
        enforce_port_ordering=False,
    )
    for route in routes:
        c.add(route.references)
    return c


if __name__ == "__main__":
    from functools import partial

    dut = partial(mzi_phase_shifter, length_y=100)

    c = add_fiber_array_optical_south_electrical_north(
        dut=dut, electrical_port_names=["top_l_e2", "top_r_e2"]
    )
    c.show(show_ports=True)
