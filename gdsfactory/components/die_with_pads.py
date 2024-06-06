import gdsfactory as gf
from gdsfactory.generic_tech import LAYER
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec


@gf.cell
def die_with_pads(
    size: tuple[float, float] = (11470.0, 4900.0),
    ngratings: int = 14,
    npads: int = 31,
    grating_pitch: float = 250.0,
    pad_pitch: float = 300.0,
    grating_coupler: ComponentSpec = "grating_coupler_te",
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad",
    layer_floorplan: LayerSpec = LAYER.FLOORPLAN,
) -> gf.Component:
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
    """
    c = gf.Component()

    fp = c << gf.c.rectangle(size=size, layer=layer_floorplan, centered=True)

    # Add optical ports
    x0 = -4925 + 2.827
    y0 = 1650

    gca = gf.c.grating_coupler_array(
        n=ngratings,
        pitch=grating_pitch,
        with_loopback=True,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
    )
    left = c << gca
    left.drotate(-90)
    left.dxmax = x0
    left.dy = fp.dy
    c.add_ports(left.ports, prefix="W")

    right = c << gca
    right.drotate(+90)
    right.dxmax = -x0
    right.dy = fp.dy
    c.add_ports(right.ports, prefix="E")

    # Add electrical ports
    x0 = -4615
    y0 = 2200
    pad = gf.get_component(pad)

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.dxmin = x0 + i * pad_pitch
        pad_ref.dymin = y0
        c.add_port(
            name=f"N{i}",
            port=pad_ref.ports["e4"],
        )

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.dxmin = x0 + i * pad_pitch
        pad_ref.dymax = -y0
        c.add_port(
            name=f"S{i}",
            port=pad_ref.ports["e2"],
        )

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = die_with_pads()
    c.show()
