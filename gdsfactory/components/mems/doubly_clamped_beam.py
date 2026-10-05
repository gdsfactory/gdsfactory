from __future__ import annotations

__all__ = ["doubly_clamped_beam"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["mems"])
def doubly_clamped_beam(
    beam_width: float = 1.0,
    beam_length: float = 30.0,
    anchor_width: float = 5.0,
    anchor_length: float = 5.0,
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """Returns a doubly clamped beam fixed at both ends.

    Two anchor pads connected by a thin beam, centered at the origin.

    Args:
        beam_width: width of the beam.
        beam_length: length of the beam between the two anchors.
        anchor_width: width (vertical) of each anchor pad.
        anchor_length: length (horizontal) of each anchor pad.
        layer: layer spec.
        port_type: port type for electrical ports.
    """
    return cf.doubly_clamped_beam(
        beam_width=beam_width,
        beam_length=beam_length,
        anchor_width=anchor_width,
        anchor_length=anchor_length,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = doubly_clamped_beam()
    c.show()
