from __future__ import annotations

__all__ = ["bent_beam"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
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
    c = Component()

    # The bend_angle is the angle between the two segments.
    # The half-angle from horizontal for each arm:
    half_angle = (180.0 - bend_angle) / 2.0
    half_angle_rad = np.radians(half_angle)

    # Segment projected lengths
    dx = beam_length * np.cos(half_angle_rad)
    dy = beam_length * np.sin(half_angle_rad)

    # Apex at (0, dy), left anchor at (-dx, 0), right anchor at (dx, 0)
    # Build the V-shape as a polygon with width

    # Build as two arms merged into one polygon

    # Left arm direction: from (-dx, 0) to (0, dy)
    left_dir = np.array([dx, dy])
    left_dir = left_dir / np.linalg.norm(left_dir)
    left_norm = np.array([-left_dir[1], left_dir[0]])  # points "up-left"

    # Right arm direction: from (0, dy) to (dx, 0)
    right_dir = np.array([dx, -dy])
    right_dir = right_dir / np.linalg.norm(right_dir)
    right_norm = np.array([-right_dir[1], right_dir[0]])  # points "up-right"

    hw = beam_width / 2

    # Left arm polygon corners
    la_start = np.array([-dx, 0.0])
    la_end = np.array([0.0, dy])

    # Right arm polygon corners
    ra_start = np.array([0.0, dy])
    ra_end = np.array([dx, 0.0])

    # Build the full V as a single polygon (outer contour going clockwise)
    poly_points = [
        tuple(la_start + left_norm * hw),
        tuple(la_end + left_norm * hw),
        tuple(ra_start + right_norm * hw),
        tuple(ra_end + right_norm * hw),
        tuple(ra_end - right_norm * hw),
        tuple(ra_start - right_norm * hw),
        tuple(la_end - left_norm * hw),
        tuple(la_start - left_norm * hw),
    ]

    c.add_polygon(poly_points, layer=layer)

    # Left anchor pad
    c.add_polygon(
        [
            (-dx - anchor_length, -anchor_width / 2),
            (-dx, -anchor_width / 2),
            (-dx, anchor_width / 2),
            (-dx - anchor_length, anchor_width / 2),
        ],
        layer=layer,
    )

    # Right anchor pad
    c.add_polygon(
        [
            (dx, -anchor_width / 2),
            (dx + anchor_length, -anchor_width / 2),
            (dx + anchor_length, anchor_width / 2),
            (dx, anchor_width / 2),
        ],
        layer=layer,
    )

    # Port at left anchor center
    c.add_port(
        "e1",
        center=(-dx - anchor_length, 0),
        width=anchor_width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )

    # Port at right anchor center
    c.add_port(
        "e2",
        center=(dx + anchor_length, 0),
        width=anchor_width,
        orientation=0,
        layer=layer,
        port_type=port_type,
    )

    return c


if __name__ == "__main__":
    c = bent_beam()
    c.show()
