from __future__ import annotations

__all__ = ["anchored_flexure"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags={"type": "mems"})
def anchored_flexure(
    hinge_width: float = 0.3,
    hinge_length: float = 5.0,
    pad_width: float = 10.0,
    pad_length: float = 10.0,
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """Returns a flexure hinge between two pads.

    Two rectangular pads connected by a thin hinge, all centered vertically.

    Args:
        hinge_width: width of the thin flexure hinge.
        hinge_length: length of the hinge connecting the two pads.
        pad_width: width (vertical) of each pad.
        pad_length: length (horizontal) of each pad.
        layer: layer spec.
        port_type: port type for electrical ports.
    """
    c = Component()

    total_length = 2 * pad_length + hinge_length
    x_start = -total_length / 2

    # Left pad
    c.add_polygon(
        [
            (x_start, -pad_width / 2),
            (x_start + pad_length, -pad_width / 2),
            (x_start + pad_length, pad_width / 2),
            (x_start, pad_width / 2),
        ],
        layer=layer,
    )

    # Thin hinge
    hinge_x0 = x_start + pad_length
    c.add_polygon(
        [
            (hinge_x0, -hinge_width / 2),
            (hinge_x0 + hinge_length, -hinge_width / 2),
            (hinge_x0 + hinge_length, hinge_width / 2),
            (hinge_x0, hinge_width / 2),
        ],
        layer=layer,
    )

    # Right pad
    right_x0 = hinge_x0 + hinge_length
    c.add_polygon(
        [
            (right_x0, -pad_width / 2),
            (right_x0 + pad_length, -pad_width / 2),
            (right_x0 + pad_length, pad_width / 2),
            (right_x0, pad_width / 2),
        ],
        layer=layer,
    )

    # Port at left pad outer edge
    c.add_port(
        "e1",
        center=(x_start, 0),
        width=pad_width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )

    # Port at right pad outer edge
    c.add_port(
        "e2",
        center=(right_x0 + pad_length, 0),
        width=pad_width,
        orientation=0,
        layer=layer,
        port_type=port_type,
    )

    return c


if __name__ == "__main__":
    c = anchored_flexure()
    c.show()
