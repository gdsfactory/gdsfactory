from __future__ import annotations

__all__ = ["bolometer"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags={"type": "mems"})
def bolometer(
    absorber_width: float = 20.0,
    absorber_length: float = 20.0,
    leg_width: float = 0.5,
    leg_length: float = 15.0,
    n_legs: int = 4,
    pad_width: float = 5.0,
    pad_length: float = 5.0,
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """Returns a bolometer thermal detector.

    A central absorber rectangle with thin L-shaped support legs extending
    outward to anchor pads distributed around the perimeter.

    Args:
        absorber_width: width of the central absorber.
        absorber_length: length of the central absorber.
        leg_width: width of each support leg.
        leg_length: length of each support leg.
        n_legs: number of support legs (distributed around perimeter).
        pad_width: width of each anchor pad.
        pad_length: length of each anchor pad.
        layer: layer spec.
        port_type: port type for electrical ports.
    """
    c = Component()

    ahw = absorber_width / 2
    ahl = absorber_length / 2

    # Central absorber rectangle centered at origin
    c.add_polygon(
        [
            (-ahl, -ahw),
            (ahl, -ahw),
            (ahl, ahw),
            (-ahl, ahw),
        ],
        layer=layer,
    )

    # Distribute legs around the absorber perimeter
    # Place legs at evenly spaced angles
    lhw = leg_width / 2

    for i in range(n_legs):
        angle = 2 * np.pi * i / n_legs
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Determine attachment point on absorber edge
        # Find which edge the ray from center at this angle hits first
        if abs(cos_a) < 1e-10:
            # Vertical ray
            attach_x = 0.0
            attach_y = ahw * np.sign(sin_a)
        elif abs(sin_a) < 1e-10:
            # Horizontal ray
            attach_x = ahl * np.sign(cos_a)
            attach_y = 0.0
        else:
            # Check intersection with vertical edges (x = +/-ahl)
            tx = ahl / abs(cos_a)
            # Check intersection with horizontal edges (y = +/-ahw)
            ty = ahw / abs(sin_a)
            if tx < ty:
                attach_x = ahl * np.sign(cos_a)
                attach_y = tx * sin_a
            else:
                attach_x = ty * cos_a
                attach_y = ahw * np.sign(sin_a)

        # L-shaped leg: first segment goes outward radially, second goes
        # along the tangential direction. For simplicity, use straight legs
        # going outward from attachment point.

        # Determine primary direction (outward from center)
        # Use the dominant axis for the leg direction
        if abs(cos_a) >= abs(sin_a):
            # Horizontal leg
            sign_x = np.sign(cos_a)
            leg_end_x = attach_x + sign_x * leg_length
            leg_end_y = attach_y

            # Horizontal leg segment
            x0 = attach_x
            x1 = leg_end_x
            if x0 > x1:
                x0, x1 = x1, x0

            c.add_polygon(
                [
                    (x0, attach_y - lhw),
                    (x1, attach_y - lhw),
                    (x1, attach_y + lhw),
                    (x0, attach_y + lhw),
                ],
                layer=layer,
            )

            # Pad at end of leg
            pad_cx = leg_end_x + sign_x * pad_length / 2
            pad_cy = leg_end_y
            c.add_polygon(
                [
                    (pad_cx - pad_length / 2, pad_cy - pad_width / 2),
                    (pad_cx + pad_length / 2, pad_cy - pad_width / 2),
                    (pad_cx + pad_length / 2, pad_cy + pad_width / 2),
                    (pad_cx - pad_length / 2, pad_cy + pad_width / 2),
                ],
                layer=layer,
            )

            # Port at outer edge of pad
            port_x = pad_cx + sign_x * pad_length / 2
            c.add_port(
                f"e{i + 1}",
                center=(port_x, pad_cy),
                width=pad_width,
                orientation=0 if sign_x > 0 else 180,
                layer=layer,
                port_type=port_type,
            )
        else:
            # Vertical leg
            sign_y = np.sign(sin_a)
            leg_end_x = attach_x
            leg_end_y = attach_y + sign_y * leg_length

            # Vertical leg segment
            y0 = attach_y
            y1 = leg_end_y
            if y0 > y1:
                y0, y1 = y1, y0

            c.add_polygon(
                [
                    (attach_x - lhw, y0),
                    (attach_x + lhw, y0),
                    (attach_x + lhw, y1),
                    (attach_x - lhw, y1),
                ],
                layer=layer,
            )

            # Pad at end of leg
            pad_cx = leg_end_x
            pad_cy = leg_end_y + sign_y * pad_width / 2
            c.add_polygon(
                [
                    (pad_cx - pad_length / 2, pad_cy - pad_width / 2),
                    (pad_cx + pad_length / 2, pad_cy - pad_width / 2),
                    (pad_cx + pad_length / 2, pad_cy + pad_width / 2),
                    (pad_cx - pad_length / 2, pad_cy + pad_width / 2),
                ],
                layer=layer,
            )

            # Port at outer edge of pad
            port_y = pad_cy + sign_y * pad_width / 2
            c.add_port(
                f"e{i + 1}",
                center=(pad_cx, port_y),
                width=pad_length,
                orientation=90 if sign_y > 0 else 270,
                layer=layer,
                port_type=port_type,
            )

    return c


if __name__ == "__main__":
    c = bolometer()
    c.show()
