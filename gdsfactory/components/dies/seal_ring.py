from __future__ import annotations

__all__ = ["seal_ring", "seal_ring_segmented"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.typings import ComponentSpec, Float2


@gf.cell_with_module_name(tags=["dies"])
def seal_ring(
    size: Float2 = (500, 500),
    seal: ComponentSpec = "via_stack",
    width: float = 10,
    padding: float = 10.0,
    with_north: bool = True,
    with_south: bool = True,
    with_east: bool = True,
    with_west: bool = True,
) -> gf.Component:
    """Returns a continuous seal ring boundary at the chip/die.

    Prevents cracks from spreading and shields when connected to ground.

    Args:
        size: of the seal.
        seal: function for the seal.
        width: of the seal.
        padding: from component to seal.
        with_north: includes seal.
        with_south: includes seal.
        with_east: includes seal.
        with_west: includes seal.
    """
    return cf.seal_ring(
        size=size,
        seal=seal,
        width=width,
        padding=padding,
        with_north=with_north,
        with_south=with_south,
        with_east=with_east,
        with_west=with_west,
    )


@gf.cell_with_module_name(tags=["dies"])
def seal_ring_segmented(
    size: Float2 = (500, 500),
    length_segment: float = 10,
    width_segment: float = 3,
    spacing_segment: float = 2,
    corner: ComponentSpec = "via_stack_corner45_extended",
    via_stack: ComponentSpec = "via_stack_m1_mtop",
    with_north: bool = True,
    with_south: bool = True,
    with_east: bool = True,
    with_west: bool = True,
) -> gf.Component:
    """Segmented Seal ring.

    Args:
        size: of the seal ring.
        length_segment: length of each segment.
        width_segment: width of each segment.
        spacing_segment: spacing between segments.
        corner: corner component.
        via_stack: via_stack component.
        with_north: includes seal.
        with_south: includes seal.
        with_east: includes seal.
        with_west: includes seal.
    """
    return cf.seal_ring_segmented(
        size=size,
        length_segment=length_segment,
        width_segment=width_segment,
        spacing_segment=spacing_segment,
        corner=corner,
        via_stack=via_stack,
        with_north=with_north,
        with_south=with_south,
        with_east=with_east,
        with_west=with_west,
    )
