from __future__ import annotations

__all__ = ["spiral_rectangular"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec
from .._schematic import spiral_schematic


@gf.cell_with_module_name
def spiral_rectangular(
    n_turns: int = 4,
    width: float = 1.0,
    start_length: float = 10.0,
    pitch: float = 3.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a rectangular/Manhattan spiral as a polygon.

    Each segment grows by pitch per half-turn, building an expanding
    rectangular spiral. The outline is created with the given width offset.

    Args:
        n_turns: number of full turns.
        width: width of the spiral trace.
        start_length: initial segment length.
        pitch: spacing between adjacent turns (center-to-center).
        layer: layer spec.
    """
    c = Component()

    # Build the center-line path of the rectangular spiral.
    # Directions cycle: +x, +y, -x, -y (right, up, left, down).
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    # Each half-turn consists of 2 segments. The segment length increases
    # by pitch every 2 segments (each half-turn).
    n_segments = n_turns * 4
    lengths = []
    length = start_length
    for i in range(n_segments):
        lengths.append(length)
        if i % 2 == 1:
            length += pitch

    # Generate center-line points.
    cx, cy = [0.0], [0.0]
    x, y = 0.0, 0.0
    for i, seg_len in enumerate(lengths):
        direction = i % 4
        x += dx[direction] * seg_len
        y += dy[direction] * seg_len
        cx.append(x)
        cy.append(y)

    # Build outer and inner offset paths.
    hw = width / 2.0
    outer_points = []
    inner_points = []

    for i in range(len(cx) - 1):
        seg_dx = cx[i + 1] - cx[i]
        seg_dy = cy[i + 1] - cy[i]
        seg_len = np.sqrt(seg_dx**2 + seg_dy**2)
        if seg_len == 0:
            continue
        # Normal direction (perpendicular, pointing left of travel).
        nx = -seg_dy / seg_len
        ny = seg_dx / seg_len

        outer_points.append((cx[i] + nx * hw, cy[i] + ny * hw))
        outer_points.append((cx[i + 1] + nx * hw, cy[i + 1] + ny * hw))
        inner_points.append((cx[i] - nx * hw, cy[i] - ny * hw))
        inner_points.append((cx[i + 1] - nx * hw, cy[i + 1] - ny * hw))

    # Close the polygon: outer forward, inner reversed.
    points = outer_points + inner_points[::-1]
    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = spiral_rectangular()
    c.show()
