__all__ = ["die_with_pads"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec, Size


@gf.cell_with_module_name(tags=["dies"])
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
    return cf.die_with_pads(
        size=size,
        ngratings=ngratings,
        npads=npads,
        grating_pitch=grating_pitch,
        pad_pitch=pad_pitch,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        pad=pad,
        layer_floorplan=layer_floorplan,
        edge_to_pad_distance=edge_to_pad_distance,
        edge_to_grating_distance=edge_to_grating_distance,
        with_loopback=with_loopback,
        loopback_radius=loopback_radius,
        pad_port_name_top=pad_port_name_top,
        pad_port_name_bot=pad_port_name_bot,
    )
