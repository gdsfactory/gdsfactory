from __future__ import annotations

__all__ = ["snspd"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec, Size


@gf.cell_with_module_name(tags=["superconductors"])
def snspd(
    wire_width: float = 0.2,
    wire_pitch: float = 0.6,
    size: Size = (10, 8),
    num_squares: int | None = None,
    turn_ratio: float = 4,
    terminals_same_side: bool = False,
    layer: LayerSpec = (1, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates an optimally-rounded SNSPD.

    Args:
        wire_width: Width of the wire.
        wire_pitch: Distance between two adjacent wires. Must be greater than `width`.
        size: Float2
            (width, height) of the rectangle formed by the outer boundary of the
            SNSPD.
        num_squares: int | None = None
            Total number of squares inside the SNSPD length.
        turn_ratio: float
            Specifies how much of the SNSPD width is dedicated to the 180 degree
            turn. A `turn_ratio` of 10 will result in 20% of the width being
            comprised of the turn.
        terminals_same_side: If True, both ports will be located on the same side of the SNSPD.
        layer: layer spec to put polygon geometry on.
        port_type: type of port to add to the component.

    """
    return cf.snspd(
        wire_width=wire_width,
        wire_pitch=wire_pitch,
        size=size,
        num_squares=num_squares,
        turn_ratio=turn_ratio,
        terminals_same_side=terminals_same_side,
        layer=layer,
        port_type=port_type,
    )
