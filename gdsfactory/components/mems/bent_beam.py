from __future__ import annotations

__all__ = ["bent_beam"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["mems"])
def bent_beam(
    beam_width: float = 1.0,
    beam_length: float = 40.0,
    bend_angle: float = 170.0,
    anchor_width: float = 5.0,
    anchor_length: float = 5.0,
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """Returns a V-shaped bent beam thermal actuator.

    Two straight beam segments meeting at an apex that points upward,
    with anchor pads at both ends.

    Args:
        beam_width: width of the beam.
        beam_length: length of each beam segment (center-line).
        bend_angle: angle between the two beam segments in degrees.
        anchor_width: width (vertical) of each anchor pad.
        anchor_length: length (horizontal) of each anchor pad.
        layer: layer spec.
        port_type: port type for electrical ports.
    """
    return cf.bent_beam(
        beam_width=beam_width,
        beam_length=beam_length,
        bend_angle=bend_angle,
        anchor_width=anchor_width,
        anchor_length=anchor_length,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = bent_beam()
    c.show()
