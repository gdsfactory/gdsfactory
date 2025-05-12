import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec, Size


@gf.cell_with_module_name
def die_with_pads(
    size: Size = (11470.0, 4900.0),
    ngratings: int = 14,
    npads: int = 31,
    grating_pitch: float = 250.0,
    pad_pitch: float = 300.0,
    grating_coupler: ComponentSpec | None = "grating_coupler_te",
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad",
    layer_floorplan: LayerSpec = "FLOORPLAN",
    edge_to_pad_distance: float = 150.0,
    edge_to_grating_distance: float = 150.0,
    with_loopback: bool = True,
    loopback_radius: float | None = None,
) -> Component:
    """A die with grating couplers and pads.

    Args:
        size: the size of the die, in um.
        ngratings: the number of grating couplers.
        npads: the number of pads.
        grating_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        grating_coupler: the grating coupler component.
        cross_section: the cross section.
        pad: the pad component.
        layer_floorplan: the layer of the floorplan.
        edge_to_pad_distance: the distance from the edge to the pads, in um.
        edge_to_grating_distance: the distance from the edge to the grating couplers, in um.
        with_loopback: if True, adds a loopback between edge GCs. Only works for rotation = 90 for now.
        loopback_radius: optional radius for loopback.
    """
    c = Component()
    fp = c << gf.c.rectangle(
        size=size, layer=layer_floorplan, centered=True, port_type=None
    )
    xs, ys = size

    # Add optical ports
    x0 = xs / 2 + edge_to_grating_distance

    if grating_coupler:
        gca = gf.c.grating_coupler_array(
            n=ngratings,
            pitch=grating_pitch,
            with_loopback=with_loopback,
            grating_coupler=grating_coupler,
            cross_section=cross_section,
            radius=loopback_radius,
        )
        left = c << gca
        left.rotate(-90)
        left.xmin = -xs / 2 + edge_to_grating_distance
        left.y = fp.y
        c.add_ports(left.ports, prefix="W")

        right = c << gca
        right.rotate(+90)
        right.xmax = xs / 2 - edge_to_grating_distance
        right.y = fp.y
        c.add_ports(right.ports, prefix="E")

    # Add electrical ports
    pad = gf.get_component(pad)
    x0 = -npads * pad_pitch / 2 + edge_to_pad_distance

    # north pads
    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymax = ys / 2 - edge_to_pad_distance
        c.add_port(
            name=f"N{i}",
            port=pad_ref.ports["e4"],
        )

    x0 = -npads * pad_pitch / 2 + edge_to_pad_distance

    # south pads
    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymin = -ys / 2 + edge_to_pad_distance
        c.add_port(
            name=f"S{i}",
            port=pad_ref.ports["e2"],
        )

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = die_with_pads(grating_coupler=None)
    c.show()
