__all__ = [
    "die_frame",
    "die_frame_phix",
    "die_frame_phix_dc",
    "die_frame_phix_rf",
    "die_frame_rf",
    "die_frame_with_pads",
]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Float2, LayerSpec, Size


@gf.cell(tags=["dies"])
def die_frame(
    size: Size = (11200.0, 5000.0),
    layer_floorplan: LayerSpec = "FLOORPLAN",
) -> gf.Component:
    """Returns a rectangular die floorplan.

    Args:
        size: die frame size (width, height), in um.
        layer_floorplan: layer for the floorplan rectangle.
    """
    return cf.die_frame(
        size=size,
        layer_floorplan=layer_floorplan,
    )


@gf.cell(tags=["dies"])
def die_frame_rf(
    size: Size = (10400.0, 5000.0),
    layer_floorplan: LayerSpec = "FLOORPLAN",
) -> gf.Component:
    """Returns a rectangular die floorplan sized for RF dies.

    Args:
        size: die frame size (width, height), in um.
        layer_floorplan: layer for the floorplan rectangle.
    """
    return cf.die_frame_rf(
        size=size,
        layer_floorplan=layer_floorplan,
    )


@gf.cell_with_module_name(tags=["dies"])
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
    return cf.die_frame_with_pads(
        die_frame=die_frame,
        ngratings=ngratings,
        npads=npads,
        grating_pitch=grating_pitch,
        pad_pitch=pad_pitch,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        pad=pad,
        edge_to_pad_distance=edge_to_pad_distance,
        edge_to_grating_distance=edge_to_grating_distance,
        with_loopback=with_loopback,
        loopback_radius=loopback_radius,
        pad_port_name_top=pad_port_name_top,
        pad_port_name_bot=pad_port_name_bot,
    )


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
    edge_to_pad_distance_left: float | None = None,
    pad_port_name_top: str = "e4",
    pad_port_name_bot: str = "e2",
    pad_port_name_rf: str = "e2",
    layer_fiducial: LayerSpec = "M3",
    fiducial_top_left: ComponentSpec | None = None,
    fiducial_top_right: ComponentSpec | None = None,
    fiducial_bottom_left: ComponentSpec | None = None,
    fiducial_bottom_right: ComponentSpec | None = None,
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
    pad_side_distance: float = 1160.0,
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
        edge_to_pad_distance_left: Optional distance from the left edge to the pads, in um. If None, uses edge_to_pad_distance for both sides.
        pad_port_name_top: name of the pad port name at the top facing south.
        pad_port_name_bot: name of the pad port name at the bottom facing north.
        pad_port_name_rf: name of the RF pad port name.
        layer_fiducial: layer for fiducials.
        fiducial_top_left: optional top-left fiducial. Defaults to the legacy
            cross on ``layer_fiducial``.
        fiducial_top_right: optional top-right fiducial. Defaults to the legacy
            circle on ``layer_fiducial``.
        fiducial_bottom_left: optional bottom-left fiducial. Defaults to the
            legacy circle on ``layer_fiducial``.
        fiducial_bottom_right: optional bottom-right fiducial. Defaults to the
            legacy circle on ``layer_fiducial``.
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
        pad_side_distance: distance from the die frame side to the first pad, in um.
        xoffset_rf_pads: RF pads x-offset.
        pad_rotation_dc_north: rotation for DC pads.
        pad_rotation_dc_south: rotation for DC pads.
        pad_rotation_rf: rotation for RF pads.
        with_loopback: if True, adds loopback structures.
    """
    return cf.die_frame_phix(
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
        edge_to_pad_distance_left=edge_to_pad_distance_left,
        pad_port_name_top=pad_port_name_top,
        pad_port_name_bot=pad_port_name_bot,
        pad_port_name_rf=pad_port_name_rf,
        layer_fiducial=layer_fiducial,
        fiducial_top_left=fiducial_top_left,
        fiducial_top_right=fiducial_top_right,
        fiducial_bottom_left=fiducial_bottom_left,
        fiducial_bottom_right=fiducial_bottom_right,
        layer_ruler=layer_ruler,
        ruler_bbox_layers=ruler_bbox_layers,
        ruler_bbox_offset=ruler_bbox_offset,
        ruler_yoffset=ruler_yoffset,
        ruler_xoffset=ruler_xoffset,
        fiber_coupler_xoffset=fiber_coupler_xoffset,
        with_right_fiber_coupler=with_right_fiber_coupler,
        with_left_fiber_coupler=with_left_fiber_coupler,
        text_offset=text_offset,
        text=text,
        pad_side_distance=pad_side_distance,
        xoffset_rf_pads=xoffset_rf_pads,
        pad_rotation_dc_north=pad_rotation_dc_north,
        pad_rotation_dc_south=pad_rotation_dc_south,
        pad_rotation_rf=pad_rotation_rf,
        with_loopback=with_loopback,
    )


@gf.cell_with_module_name(tags=["dies"])
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
    fiducial_top_left: ComponentSpec | None = None,
    fiducial_top_right: ComponentSpec | None = None,
    fiducial_bottom_left: ComponentSpec | None = None,
    fiducial_bottom_right: ComponentSpec | None = None,
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
    pad_side_distance: float = 1160.0,
) -> Component:
    """A PHIX die frame with DC pads only.

    Args:
        die_frame: die_frame spec.
        nfibers: the number of grating couplers.
        npads: the number of pads.
        npads_rf: the number of RF pads on the left side.
        fiber_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        pad_pitch_gsg: the pitch of the GSG pads, in um.
        edge_coupler: the edge coupler component.
        grating_coupler: Optional grating coupler.
        cross_section: the cross section.
        pad: the pad component.
        pad_gsg: the GSG pad component.
        edge_to_pad_distance: the distance from the edge to the pads, in um.
        pad_port_name_top: name of the pad port name at the top facing south.
        pad_port_name_bot: name of the pad port name at the bottom facing north.
        layer_fiducial: layer for fiducials.
        fiducial_top_left: optional top-left fiducial; defaults to a cross.
        fiducial_top_right: optional top-right fiducial; defaults to a circle.
        fiducial_bottom_left: optional bottom-left fiducial; defaults to a circle.
        fiducial_bottom_right: optional bottom-right fiducial; defaults to a circle.
        layer_ruler: layer for ruler.
        ruler_bbox_layers: layers for bbox.
        ruler_bbox_offset: offset for bbox.
        ruler_yoffset: y-offset for ruler.
        ruler_xoffset: x-offset for ruler.
        with_right_fiber_coupler: if True, adds edge couplers on the right side.
        with_left_fiber_coupler: if True, adds edge couplers on the left side.
        fiber_coupler_xoffset: x-offset for fiber couplers.
        text_offset: offset for text.
        text: text component spec.
        pad_rotation_dc_north: rotation for DC pads.
        pad_rotation_dc_south: rotation for DC pads.
        pad_side_distance: distance from the die frame side to the first pad, in um.
    """
    return cf.die_frame_phix_dc(
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
        fiducial_top_left=fiducial_top_left,
        fiducial_top_right=fiducial_top_right,
        fiducial_bottom_left=fiducial_bottom_left,
        fiducial_bottom_right=fiducial_bottom_right,
        layer_ruler=layer_ruler,
        ruler_bbox_layers=ruler_bbox_layers,
        ruler_bbox_offset=ruler_bbox_offset,
        ruler_yoffset=ruler_yoffset,
        ruler_xoffset=ruler_xoffset,
        with_right_fiber_coupler=with_right_fiber_coupler,
        with_left_fiber_coupler=with_left_fiber_coupler,
        fiber_coupler_xoffset=fiber_coupler_xoffset,
        text_offset=text_offset,
        text=text,
        pad_rotation_dc_north=pad_rotation_dc_north,
        pad_rotation_dc_south=pad_rotation_dc_south,
        pad_side_distance=pad_side_distance,
    )


@gf.cell_with_module_name(tags=["dies"])
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
    fiducial_top_left: ComponentSpec | None = None,
    fiducial_top_right: ComponentSpec | None = None,
    fiducial_bottom_left: ComponentSpec | None = None,
    fiducial_bottom_right: ComponentSpec | None = None,
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
    pad_side_distance: float = 350.0,
    xoffset_rf_pads: float = 50,
    pad_rotation_rf: float = 0,
    pad_rotation_dc_north: float = 0,
    pad_rotation_dc_south: float = 0,
) -> Component:
    """A PHIX die frame with DC and RF pads.

    Args:
        die_frame: die_frame spec.
        nfibers: the number of grating couplers.
        npads: the number of pads.
        npads_rf: the number of RF pads on the left side.
        fiber_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        pad_pitch_gsg: the pitch of the GSG pads, in um.
        edge_coupler: the edge coupler component.
        grating_coupler: Optional grating coupler.
        cross_section: the cross section.
        pad: the pad component.
        pad_gsg: the GSG pad component.
        edge_to_pad_distance: the distance from the edge to the pads, in um.
        pad_port_name_top: name of the pad port name at the top facing south.
        pad_port_name_bot: name of the pad port name at the bottom facing north.
        pad_port_name_rf: name of the RF pad port name.
        layer_fiducial: layer for fiducials.
        fiducial_top_left: optional top-left fiducial; defaults to a cross.
        fiducial_top_right: optional top-right fiducial; defaults to a circle.
        fiducial_bottom_left: optional bottom-left fiducial; defaults to a circle.
        fiducial_bottom_right: optional bottom-right fiducial; defaults to a circle.
        layer_ruler: layer for ruler.
        ruler_bbox_layers: layers for bbox.
        ruler_bbox_offset: offset for bbox.
        ruler_yoffset: y-offset for ruler.
        ruler_xoffset: x-offset for ruler.
        with_right_fiber_coupler: if True, adds edge couplers on the right side.
        with_left_fiber_coupler: if True, adds edge couplers on the left side.
        fiber_coupler_xoffset: x-offset for fiber couplers.
        text_offset: offset for text.
        text: text component spec.
        pad_side_distance: distance from the die frame side to the first pad, in um.
        xoffset_rf_pads: RF pads x-offset.
        pad_rotation_rf: rotation for RF pads.
        pad_rotation_dc_north: rotation for DC pads.
        pad_rotation_dc_south: rotation for DC pads.
    """
    return cf.die_frame_phix_rf(
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
        fiducial_top_left=fiducial_top_left,
        fiducial_top_right=fiducial_top_right,
        fiducial_bottom_left=fiducial_bottom_left,
        fiducial_bottom_right=fiducial_bottom_right,
        layer_ruler=layer_ruler,
        ruler_bbox_layers=ruler_bbox_layers,
        ruler_bbox_offset=ruler_bbox_offset,
        ruler_yoffset=ruler_yoffset,
        ruler_xoffset=ruler_xoffset,
        with_right_fiber_coupler=with_right_fiber_coupler,
        with_left_fiber_coupler=with_left_fiber_coupler,
        fiber_coupler_xoffset=fiber_coupler_xoffset,
        text_offset=text_offset,
        text=text,
        pad_side_distance=pad_side_distance,
        xoffset_rf_pads=xoffset_rf_pads,
        pad_rotation_rf=pad_rotation_rf,
        pad_rotation_dc_north=pad_rotation_dc_north,
        pad_rotation_dc_south=pad_rotation_dc_south,
    )


if __name__ == "__main__":
    # text_m3 = partial(gf.c.text_rectangular, layer="M3", size=20)
    text_m3 = None
    edge_coupler = CellAlias(gf.c.edge_coupler_silicon, length=200)
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
