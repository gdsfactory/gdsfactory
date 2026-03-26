from __future__ import annotations

__all__ = ["folded_spring"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def folded_spring(
    beam_width: float = 0.5,
    beam_length: float = 20.0,
    n_folds: int = 4,
    fold_gap: float = 1.0,
    anchor_width: float = 5.0,
    anchor_length: float = 3.0,
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """Returns a folded flexure spring (serpentine meander).

    Alternating horizontal beams connected at their ends forming a
    serpentine pattern. Starts at a bottom anchor and ends at a top anchor.

    Args:
        beam_width: width of each beam segment.
        beam_length: length of each horizontal beam segment.
        n_folds: number of horizontal beam segments.
        fold_gap: vertical gap between adjacent beams.
        anchor_width: width (horizontal) of the anchor pads.
        anchor_length: length (vertical) of the anchor pads.
        layer: layer spec.
        port_type: port type for electrical ports.
    """
    c = Component()

    # Bottom anchor, centered horizontally at x=0
    anchor_bottom_y = -anchor_length
    c.add_polygon(
        [
            (-anchor_width / 2, anchor_bottom_y),
            (anchor_width / 2, anchor_bottom_y),
            (anchor_width / 2, 0),
            (-anchor_width / 2, 0),
        ],
        layer=layer,
    )

    # Draw serpentine beams
    # Each beam is horizontal. Even-indexed beams go from x=0 to x=beam_length,
    # odd-indexed beams go from x=0 to x=beam_length as well, but connected
    # on alternating sides.
    y_cursor = 0.0

    for i in range(n_folds):
        y_bottom = y_cursor
        y_top = y_cursor + beam_width

        if i % 2 == 0:
            # Beam extends to the right from x=0
            c.add_polygon(
                [
                    (0, y_bottom),
                    (beam_length, y_bottom),
                    (beam_length, y_top),
                    (0, y_top),
                ],
                layer=layer,
            )
        else:
            # Beam extends to the left from x=beam_length
            c.add_polygon(
                [
                    (0, y_bottom),
                    (beam_length, y_bottom),
                    (beam_length, y_top),
                    (0, y_top),
                ],
                layer=layer,
            )

        # Add connecting segment at the end to the next beam
        if i < n_folds - 1:
            conn_y_bottom = y_top
            conn_y_top = y_top + fold_gap

            if i % 2 == 0:
                # Connect on the right side
                c.add_polygon(
                    [
                        (beam_length - beam_width, conn_y_bottom),
                        (beam_length, conn_y_bottom),
                        (beam_length, conn_y_top),
                        (beam_length - beam_width, conn_y_top),
                    ],
                    layer=layer,
                )
            else:
                # Connect on the left side
                c.add_polygon(
                    [
                        (0, conn_y_bottom),
                        (beam_width, conn_y_bottom),
                        (beam_width, conn_y_top),
                        (0, conn_y_top),
                    ],
                    layer=layer,
                )

        y_cursor += beam_width + fold_gap

    # Top anchor
    top_y = y_cursor - fold_gap  # top of last beam
    # Determine x position of top anchor based on last beam's exit side
    if (n_folds - 1) % 2 == 0:
        # Last beam exits on right side (x=beam_length)
        anchor_top_x = beam_length
    else:
        # Last beam exits on left side (x=0)
        anchor_top_x = 0.0

    c.add_polygon(
        [
            (anchor_top_x - anchor_width / 2, top_y),
            (anchor_top_x + anchor_width / 2, top_y),
            (anchor_top_x + anchor_width / 2, top_y + anchor_length),
            (anchor_top_x - anchor_width / 2, top_y + anchor_length),
        ],
        layer=layer,
    )

    # Port at bottom anchor
    c.add_port(
        "e1",
        center=(0, anchor_bottom_y),
        width=anchor_width,
        orientation=270,
        layer=layer,
        port_type=port_type,
    )

    # Port at top anchor
    c.add_port(
        "e2",
        center=(anchor_top_x, top_y + anchor_length),
        width=anchor_width,
        orientation=90,
        layer=layer,
        port_type=port_type,
    )

    return c


if __name__ == "__main__":
    c = folded_spring()
    c.show()
