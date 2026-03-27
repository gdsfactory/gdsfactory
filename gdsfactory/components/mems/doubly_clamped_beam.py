from __future__ import annotations

__all__ = ["doubly_clamped_beam"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
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
    c = Component()

    total_length = 2 * anchor_length + beam_length
    x_start = -total_length / 2

    # Left anchor
    c.add_polygon(
        [
            (x_start, -anchor_width / 2),
            (x_start + anchor_length, -anchor_width / 2),
            (x_start + anchor_length, anchor_width / 2),
            (x_start, anchor_width / 2),
        ],
        layer=layer,
    )

    # Central beam
    beam_x0 = x_start + anchor_length
    c.add_polygon(
        [
            (beam_x0, -beam_width / 2),
            (beam_x0 + beam_length, -beam_width / 2),
            (beam_x0 + beam_length, beam_width / 2),
            (beam_x0, beam_width / 2),
        ],
        layer=layer,
    )

    # Right anchor
    right_x0 = beam_x0 + beam_length
    c.add_polygon(
        [
            (right_x0, -anchor_width / 2),
            (right_x0 + anchor_length, -anchor_width / 2),
            (right_x0 + anchor_length, anchor_width / 2),
            (right_x0, anchor_width / 2),
        ],
        layer=layer,
    )

    # Port at left anchor outside edge
    c.add_port(
        "e1",
        center=(x_start, 0),
        width=anchor_width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )

    # Port at right anchor outside edge
    c.add_port(
        "e2",
        center=(right_x0 + anchor_length, 0),
        width=anchor_width,
        orientation=0,
        layer=layer,
        port_type=port_type,
    )

    return c


if __name__ == "__main__":
    c = doubly_clamped_beam()
    c.show()
