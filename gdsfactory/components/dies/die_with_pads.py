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
    pad_port_name_top: str = "e4",
    pad_port_name_bot: str = "e2",
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
        pad_port_name_top: name of the pad port name at the btop facing south.
        pad_port_name_bot: name of the pad port name at the bottom facing north.
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
            port=pad_ref.ports[pad_port_name_top],
        )

    x0 = -npads * pad_pitch / 2 + edge_to_pad_distance

    # south pads
    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymin = -ys / 2 + edge_to_pad_distance
        c.add_port(
            name=f"S{i}",
            port=pad_ref.ports[pad_port_name_bot],
        )

    c.auto_rename_ports()
    return c


@gf.cell_with_module_name
def die_with_pads_gsg(
    size: Size = (11470.0, 4900.0),
    ngratings: int = 14,
    npads: int = 31,
    grating_pitch: float = 250.0,
    pad_pitch: float = 300.0,
    grating_coupler: ComponentSpec | None = "grating_coupler_te",
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad_gsg",
    layer_floorplan: LayerSpec = "FLOORPLAN",
    edge_to_pad_distance: float = 150.0,
    edge_to_grating_distance: float = 150.0,
    with_loopback: bool = True,
    loopback_radius: float | None = None,
    pad_rotation: float = 90,
    pad_port_name_top: str = "e1",
    pad_port_name_bot: str = "e1",
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
        pad_rotation: rotation angle of the pads.
        pad_port_name_top: name of the pad port name at the btop facing south.
        pad_port_name_bot: name of the pad port name at the bottom facing north.
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
        pad_ref.rotate(+pad_rotation)
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymax = ys / 2 - edge_to_pad_distance
        c.add_port(
            name=f"N{i}",
            port=pad_ref.ports[pad_port_name_top],
        )

    x0 = -npads * pad_pitch / 2 + edge_to_pad_distance

    # south pads
    for i in range(npads):
        pad_ref = c << pad
        pad_ref.rotate(-pad_rotation)
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymin = -ys / 2 + edge_to_pad_distance
        c.add_port(
            name=f"S{i}",
            port=pad_ref.ports[pad_port_name_bot],
        )

    c.auto_rename_ports()
    return c


@gf.cell_with_module_name
def die_with_pads_phix(
    size: Size = (10570.0, 5000.0),
    ngratings: int = 14,
    npads: int = 59,
    fiber_pitch: float = 127.0,
    pad_pitch: float = 150.0,
    grating_coupler: ComponentSpec | None = "grating_coupler_te",
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad",
    layer_floorplan: LayerSpec = "FLOORPLAN",
    edge_to_pad_distance: float = 200.0,
    edge_to_grating_distance: float = 150.0,
    with_loopback: bool = True,
    loopback_radius: float | None = None,
    pad_port_name_top: str = "e4",
    pad_port_name_bot: str = "e2",
    layer_fiducial: LayerSpec = "M3",
    layer_ruler: LayerSpec = "M3",
    with_right_edge_coupler: bool = True,
    with_left_edge_coupler: bool = True,
) -> Component:
    """A die with grating couplers and pads.

    Args:
        size: the size of the die, in um.
        ngratings: the number of grating couplers.
        npads: the number of pads.
        fiber_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        grating_coupler: the grating coupler component.
        cross_section: the cross section.
        pad: the pad component.
        layer_floorplan: the layer of the floorplan.
        edge_to_pad_distance: the distance from the edge to the pads, in um.
        edge_to_grating_distance: the distance from the edge to the grating couplers, in um.
        with_loopback: if True, adds a loopback between edge GCs. Only works for rotation = 90 for now.
        loopback_radius: optional radius for loopback.
        pad_port_name_top: name of the pad port name at the btop facing south.
        pad_port_name_bot: name of the pad port name at the bottom facing north.
        layer_fiducial: layer for fiducials.
        layer_ruler: layer for ruler.
        with_right_edge_coupler: if True, adds edge couplers on the right side.
        with_left_edge_coupler: if True, adds edge couplers on the left side.
    """
    c = Component()
    fp = c << gf.c.rectangle(
        size=size, layer=layer_floorplan, centered=True, port_type=None
    )
    xs, ys = size

    # Add optical ports
    x0 = xs / 2 + edge_to_grating_distance

    if grating_coupler and not with_right_edge_coupler:
        gca = gf.c.grating_coupler_array(
            n=ngratings,
            pitch=fiber_pitch,
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

    if with_right_edge_coupler:
        ruler_top_right = c << gf.c.litho_ruler_staircase(layer=layer_ruler)
        ruler_top_right.xmax = fp.xmax
        ruler_top_right.ymax = fp.ymax - 300

        ruler_bot_right = c << gf.c.litho_ruler_staircase(layer=layer_ruler)
        ruler_bot_right.xmax = fp.xmax
        ruler_bot_right.ymin = fp.ymin + 300

    if with_left_edge_coupler:
        ruler_top_left = c << gf.c.litho_ruler_staircase(layer=layer_ruler)
        ruler_top_left.rotate(180)
        ruler_top_left.xmin = fp.xmin
        ruler_top_left.ymax = fp.ymax - 300

        ruler_bot_left = c << gf.c.litho_ruler_staircase(layer=layer_ruler)
        ruler_bot_left.rotate(180)
        ruler_bot_left.xmin = fp.xmin
        ruler_bot_left.ymin = fp.ymin + 300

    # Add electrical ports
    pad = gf.get_component(pad)

    x0_pads = -npads * pad_pitch / 2 + edge_to_pad_distance - 490
    x0 = x0_pads

    top_left = c << gf.c.cross(layer=layer_fiducial, length=150, width=20)
    top_left.xmax = x0 - 75
    top_left.y = +ys / 2 - edge_to_pad_distance - 50

    # north pads
    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymax = ys / 2 - edge_to_pad_distance
        c.add_port(
            name=f"N{i}",
            port=pad_ref.ports[pad_port_name_top],
        )
    top_right = c << gf.c.circle(layer=layer_fiducial, radius=75)
    top_right.xmin = pad_ref.xmax + 480
    top_right.y = +ys / 2 - edge_to_pad_distance - 50

    bot_left = c << gf.c.circle(layer=layer_fiducial, radius=75)
    bot_left.xmax = x0 - 75
    bot_left.y = -ys / 2 + edge_to_pad_distance + 50

    x0 = x0_pads
    # south pads
    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymin = -ys / 2 + edge_to_pad_distance
        c.add_port(
            name=f"S{i}",
            port=pad_ref.ports[pad_port_name_bot],
        )

    bot_right = c << gf.c.circle(layer=layer_fiducial, radius=75)
    bot_right.xmin = pad_ref.xmax + 480
    bot_right.ymin = -ys / 2 + edge_to_pad_distance
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = die_with_pads_phix()
    c.show()
