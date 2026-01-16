__all__ = [
    "die_frame",
    "die_frame_phix",
    "die_frame_phix_dc",
    "die_frame_phix_rf",
    "die_frame_rf",
    "die_frame_with_pads",
]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Float2, LayerSpec, Size


@gf.cell
def die_frame(
    size: Size = (11200.0, 5000.0),
    layer_floorplan: LayerSpec = "FLOORPLAN",
) -> gf.Component:
    return gf.c.rectangle(
        size=size, layer=layer_floorplan, centered=True, port_type=None
    )


@gf.cell
def die_frame_rf(
    size: Size = (10400.0, 5000.0),
    layer_floorplan: LayerSpec = "FLOORPLAN",
) -> gf.Component:
    return gf.c.rectangle(
        size=size, layer=layer_floorplan, centered=True, port_type=None
    )


@gf.cell_with_module_name
def die_frame_with_pads(
    die_frame: ComponentSpec = "die_frame",
    ngratings: int = 14,
    npads: int = 31,
    grating_pitch: float = 250.0,
    pad_pitch: float = 300.0,
    grating_coupler: ComponentSpec | None = "grating_coupler_te",
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad",
    edge_to_pad_distance: float = 150.0,
    edge_to_grating_distance: float = 150.0,
    with_loopback: bool = True,
    loopback_radius: float | None = None,
    pad_port_name_top: str = "e4",
    pad_port_name_bot: str = "e2",
) -> Component:
    """A die_frame with grating couplers and pads.

    Args:
        die_frame: die_frame spec.
        ngratings: the number of grating couplers.
        npads: the number of pads.
        grating_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        grating_coupler: the grating coupler component.
        cross_section: the cross section.
        pad: the pad component.
        edge_to_pad_distance: the distance from the edge to the pads, in um.
        edge_to_grating_distance: the distance from the edge to the grating couplers, in um.
        with_loopback: if True, adds a loopback between edge GCs. Only works for rotation = 90 for now.
        loopback_radius: optional radius for loopback.
        pad_port_name_top: name of the pad port name at the btop facing south.
        pad_port_name_bot: name of the pad port name at the bottom facing north.
    """
    c = Component()

    d = gf.get_component(die_frame)
    fp = c << d
    fp.x = 0
    fp.y = 0
    xs, ys = fp.xsize, fp.ysize

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


def die_frame_phix(
    die_frame: ComponentSpec = "die_frame",
    nfibers: int = 32,
    npads: int = 60,
    npads_rf: int = 6,
    fiber_pitch: float = 127.0,
    pad_pitch: float = 150.0,
    pad_pitch_gsg: float = 720.0,
    edge_coupler: ComponentSpec | None = "edge_coupler_silicon",
    grating_coupler: ComponentSpec | None = None,
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad",
    pad_gsg: ComponentSpec = "pad_gsg",
    edge_to_pad_distance: float = 200.0,
    pad_port_name_top: str = "e4",
    pad_port_name_bot: str = "e2",
    pad_port_name_rf: str = "e2",
    layer_fiducial: LayerSpec = "M3",
    layer_ruler: LayerSpec = "WG",
    ruler_bbox_layers: tuple[LayerSpec, ...] | None = None,
    ruler_bbox_offset: float = 3.0,
    ruler_yoffset: float = 0,
    ruler_xoffset: float = 0,
    fiber_coupler_xoffset: float = 0,
    with_right_fiber_coupler: bool = True,
    with_left_fiber_coupler: bool = True,
    text_offset: Float2 = (20, 10),
    text: ComponentSpec | None = "text_rectangular",
    xoffset_dc_pads: float = -100,
    xoffset_rf_pads: float = 50,
    pad_rotation_dc_north: float = 0,
    pad_rotation_dc_south: float = 0,
    pad_rotation_rf: float = 0,
    with_loopback: bool = True,
) -> Component:
    """A die_frame with grating couplers and pads.

    Args:
        die_frame: die_frame spec.
        nfibers: the number of grating couplers.
        npads: the number of pads.
        npads_rf: the number of RF pads on the left side.
        fiber_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        pad_pitch_gsg: the pitch of the GSG pads, in um.
        edge_coupler: the grating coupler component.
        grating_coupler: Optional grating coupler.
        cross_section: the cross section.
        pad: the pad component.
        pad_gsg: the GSG pad component.
        edge_to_pad_distance: the distance from the edge to the pads, in um.
        pad_port_name_top: name of the pad port name at the top facing south.
        pad_port_name_bot: name of the pad port name at the bottom facing north.
        pad_port_name_rf: name of the RF pad port name.
        layer_fiducial: layer for fiducials.
        layer_ruler: layer for ruler.
        ruler_bbox_layers: layers for bbox.
        ruler_bbox_offset: offset for bbox.
        ruler_yoffset: y-offset for ruler.
        ruler_xoffset: x-offset for ruler.
        fiber_coupler_xoffset: x-offset for fiber couplers.
        with_right_fiber_coupler: if True, adds edge couplers on the right side.
        with_left_fiber_coupler: if True, adds edge couplers on the left side.
        text_offset: offset for text.
        text: text component spec.
        xoffset_dc_pads: DC pads x-offset.
        xoffset_rf_pads: RF pads x-offset.
        pad_rotation_dc_north: rotation for DC pads.
        pad_rotation_dc_south: rotation for DC pads.
        pad_rotation_rf: rotation for RF pads.
    """
    if npads > 60:
        raise ValueError("npads should be <= 60. Reach out to PHIX for support.")

    c = Component()

    d = gf.get_component(die_frame)
    fp = c << d
    fp.x = 0
    fp.y = 0
    xs, ys = fp.xsize, fp.ysize

    # Add optical ports
    x0 = xs / 2

    if edge_coupler or grating_coupler:
        if edge_coupler:
            if with_loopback:
                gca = gf.c.edge_coupler_array_with_loopback(
                    n=nfibers,
                    pitch=fiber_pitch,
                    edge_coupler=edge_coupler,
                    cross_section=cross_section,
                    text_offset=text_offset,
                    text=text,
                    x_reflection=False,
                )
                gca_left = gf.c.edge_coupler_array_with_loopback(
                    n=nfibers,
                    pitch=fiber_pitch,
                    edge_coupler=edge_coupler,
                    cross_section=cross_section,
                    text_offset=(-text_offset[0], text_offset[1]),
                    text=text,
                    x_reflection=True,
                )
            else:
                gca = gf.c.edge_coupler_array(
                    n=nfibers,
                    pitch=fiber_pitch,
                    edge_coupler=edge_coupler,
                    text_offset=text_offset,
                    text=text,
                    x_reflection=False,
                )
                gca_left = gf.c.edge_coupler_array(
                    n=nfibers,
                    pitch=fiber_pitch,
                    edge_coupler=edge_coupler,
                    text_offset=(-text_offset[0], text_offset[1]),
                    text=text,
                    x_reflection=True,
                )

            if with_left_fiber_coupler:
                left = c << gca_left
                left.xmin = -xs / 2 - fiber_coupler_xoffset
                left.y = fp.y
                c.add_ports(left.ports, prefix="W")

            if with_right_fiber_coupler:
                right = c << gca
                right.xmax = xs / 2 + fiber_coupler_xoffset
                right.y = fp.y
                c.add_ports(right.ports, prefix="E")

        else:
            gca = gf.c.grating_coupler_array(
                n=nfibers,
                pitch=fiber_pitch,
                cross_section=cross_section,
                with_loopback=True,
            )
            gca_left = gf.c.grating_coupler_array(
                n=nfibers,
                pitch=fiber_pitch,
                cross_section=cross_section,
                with_loopback=True,
            )
            fiber_coupler_xoffset -= 750

            if with_left_fiber_coupler:
                left = c << gca_left
                left.rotate(-90)
                left.xmin = -xs / 2 - fiber_coupler_xoffset
                left.y = fp.y
                c.add_ports(left.ports, prefix="W")

            if with_right_fiber_coupler:
                right = c << gca
                right.rotate(+90)
                right.xmax = xs / 2 + fiber_coupler_xoffset
                right.y = fp.y
                c.add_ports(right.ports, prefix="E")
    ruler = gf.c.ruler(
        layer=layer_ruler,
        bbox_layers=ruler_bbox_layers,
        bbox_offset=ruler_bbox_offset,
    )

    if with_right_fiber_coupler:
        ruler_top_right = c << ruler
        ruler_top_right.xmax = fp.xmax - ruler_xoffset
        ruler_top_right.ymax = fp.ymax - 300 + ruler_yoffset

        ruler_bot_right = c << ruler
        ruler_bot_right.xmax = fp.xmax - ruler_xoffset
        ruler_bot_right.ymin = fp.ymin + 300 - ruler_yoffset

    if with_left_fiber_coupler:
        ruler_top_left = c << ruler
        ruler_top_left.rotate(180)
        ruler_top_left.xmin = fp.xmin + ruler_xoffset
        ruler_top_left.ymax = fp.ymax - 300 + ruler_yoffset

        ruler_bot_left = c << ruler
        ruler_bot_left.rotate(180)
        ruler_bot_left.xmin = fp.xmin + ruler_xoffset
        ruler_bot_left.ymin = fp.ymin + 300 - ruler_yoffset

    else:
        # left RF pads
        y0 = fp.ymax - 390 - pad_pitch_gsg / 2 + 50
        for i in range(npads_rf):
            pad_ref = c << gf.get_component(pad_gsg)
            pad_ref.rotate(pad_rotation_rf)
            pad_ref.y = y0 - i * pad_pitch_gsg
            pad_ref.xmin = fp.xmin + xoffset_rf_pads
            c.add_port(
                name=f"e{i}",
                port=pad_ref.ports[pad_port_name_rf],
            )

    # Add electrical ports
    pad = gf.get_component(pad)

    x0_pads = (
        -npads * pad_pitch / 2 + edge_to_pad_distance - pad_pitch / 2 + xoffset_dc_pads
    )
    x0 = x0_pads

    top_left = c << gf.c.cross(layer=layer_fiducial, length=150, width=20)
    top_left.xmax = x0 - 75
    top_left.y = +ys / 2 - edge_to_pad_distance - 50

    # north pads
    for i in range(npads):
        pad_ref = c << pad
        pad_ref.rotate(pad_rotation_dc_north)
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
        pad_ref.rotate(pad_rotation_dc_south)
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


@gf.cell_with_module_name
def die_frame_phix_dc(
    die_frame: ComponentSpec = "die_frame",
    nfibers: int = 32,
    npads: int = 59,
    npads_rf: int = 6,
    fiber_pitch: float = 127.0,
    pad_pitch: float = 150.0,
    pad_pitch_gsg: float = 720.0,
    edge_coupler: ComponentSpec | None = "edge_coupler_silicon",
    grating_coupler: ComponentSpec | None = None,
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad",
    pad_gsg: ComponentSpec = "pad_gsg",
    edge_to_pad_distance: float = 200.0,
    pad_port_name_top: str = "e4",
    pad_port_name_bot: str = "e2",
    layer_fiducial: LayerSpec = "M3",
    layer_ruler: LayerSpec = "WG",
    ruler_bbox_layers: tuple[LayerSpec, ...] | None = None,
    ruler_bbox_offset: float = 3.0,
    ruler_yoffset: float = 0,
    ruler_xoffset: float = 0,
    with_right_fiber_coupler: bool = True,
    with_left_fiber_coupler: bool = True,
    fiber_coupler_xoffset: float = 0,
    text_offset: Float2 = (20, 10),
    text: ComponentSpec | None = None,
    pad_rotation_dc_north: float = 0,
    pad_rotation_dc_south: float = 0,
) -> Component:
    return die_frame_phix(
        die_frame=die_frame,
        nfibers=nfibers,
        npads=npads,
        npads_rf=npads_rf,
        fiber_pitch=fiber_pitch,
        pad_pitch=pad_pitch,
        pad_pitch_gsg=pad_pitch_gsg,
        edge_coupler=edge_coupler,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        pad=pad,
        pad_gsg=pad_gsg,
        edge_to_pad_distance=edge_to_pad_distance,
        pad_port_name_top=pad_port_name_top,
        pad_port_name_bot=pad_port_name_bot,
        layer_fiducial=layer_fiducial,
        layer_ruler=layer_ruler,
        ruler_bbox_layers=ruler_bbox_layers,
        ruler_bbox_offset=ruler_bbox_offset,
        ruler_yoffset=ruler_yoffset,
        ruler_xoffset=ruler_xoffset,
        with_right_fiber_coupler=with_right_fiber_coupler,
        with_left_fiber_coupler=with_left_fiber_coupler,
        text_offset=text_offset,
        text=text,
        fiber_coupler_xoffset=fiber_coupler_xoffset,
        pad_rotation_dc_north=pad_rotation_dc_north,
        pad_rotation_dc_south=pad_rotation_dc_south,
    )


@gf.cell_with_module_name
def die_frame_phix_rf(
    die_frame: ComponentSpec = "die_frame_rf",
    nfibers: int = 32,
    npads: int = 59,
    npads_rf: int = 6,
    fiber_pitch: float = 127.0,
    pad_pitch: float = 150.0,
    pad_pitch_gsg: float = 720.0,
    edge_coupler: ComponentSpec | None = "edge_coupler_silicon",
    grating_coupler: ComponentSpec | None = None,
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad",
    pad_gsg: ComponentSpec = "pad_gsg",
    edge_to_pad_distance: float = 200.0,
    pad_port_name_top: str = "e4",
    pad_port_name_bot: str = "e2",
    pad_port_name_rf: str = "e2",
    layer_fiducial: LayerSpec = "M3",
    layer_ruler: LayerSpec = "WG",
    ruler_bbox_layers: tuple[LayerSpec, ...] | None = None,
    ruler_bbox_offset: float = 3.0,
    ruler_yoffset: float = 0,
    ruler_xoffset: float = 0,
    with_right_fiber_coupler: bool = True,
    with_left_fiber_coupler: bool = False,
    fiber_coupler_xoffset: float = 0,
    text_offset: Float2 = (20, 10),
    text: ComponentSpec | None = None,
    xoffset_dc_pads: float = -500,
    xoffset_rf_pads: float = 50,
    pad_rotation_rf: float = 0,
    pad_rotation_dc_north: float = 0,
    pad_rotation_dc_south: float = 0,
) -> Component:
    return die_frame_phix(
        die_frame=die_frame,
        nfibers=nfibers,
        npads=npads,
        npads_rf=npads_rf,
        fiber_pitch=fiber_pitch,
        pad_pitch=pad_pitch,
        pad_pitch_gsg=pad_pitch_gsg,
        edge_coupler=edge_coupler,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        pad=pad,
        pad_gsg=pad_gsg,
        edge_to_pad_distance=edge_to_pad_distance,
        pad_port_name_top=pad_port_name_top,
        pad_port_name_bot=pad_port_name_bot,
        pad_port_name_rf=pad_port_name_rf,
        layer_fiducial=layer_fiducial,
        layer_ruler=layer_ruler,
        ruler_bbox_layers=ruler_bbox_layers,
        ruler_bbox_offset=ruler_bbox_offset,
        ruler_yoffset=ruler_yoffset,
        ruler_xoffset=ruler_xoffset,
        with_right_fiber_coupler=with_right_fiber_coupler,
        with_left_fiber_coupler=with_left_fiber_coupler,
        text_offset=text_offset,
        text=text,
        xoffset_dc_pads=xoffset_dc_pads,
        fiber_coupler_xoffset=fiber_coupler_xoffset,
        xoffset_rf_pads=xoffset_rf_pads,
        pad_rotation_dc_south=pad_rotation_dc_south,
        pad_rotation_dc_north=pad_rotation_dc_north,
        pad_rotation_rf=pad_rotation_rf,
    )


if __name__ == "__main__":
    from functools import partial

    # text_m3 = partial(gf.c.text_rectangular, layer="M3", size=20)
    text_m3 = None
    edge_coupler = partial(gf.c.edge_coupler_silicon, length=200)
    grating_coupler = "grating_coupler_te"

    c = die_frame_phix_dc(edge_coupler=edge_coupler, text=text_m3)
    c.write_gds("/Users/j/Downloads/die_frame_phix_dc.gds")

    grating_coupler = "grating_coupler_te"
    c = die_frame_phix_dc(
        die_frame=die_frame(size=(11800, 5000)),
        edge_coupler=None,
        text=text_m3,
        grating_coupler=grating_coupler,
    )
    c.write_gds("/Users/j/Downloads/die_frame_phix_rf_grating_coupler.gds")

    # c = die_frame_phix_rf(edge_coupler=edge_coupler)
    # c.write_gds("/Users/j/Downloads/die_frame_phix_rf.gds")
    c.show()
