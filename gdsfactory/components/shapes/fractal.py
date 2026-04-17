from __future__ import annotations

__all__ = ["fractal"]

from typing import Literal

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


def _sierpinski_triangle(depth: int, size: float) -> list[list[tuple[float, float]]]:
    """Generate Sierpinski triangle polygons recursively."""
    h = size * np.sqrt(3) / 2
    base_tri = [(-size / 2, -h / 3), (size / 2, -h / 3), (0, 2 * h / 3)]

    if depth == 0:
        return [base_tri]

    polygons: list[list[tuple[float, float]]] = []
    scale = 0.5
    offsets = [
        (0, h / 3),
        (-size / 4, -h / 6),
        (size / 4, -h / 6),
    ]
    sub = _sierpinski_triangle(depth - 1, size * scale)
    for ox, oy in offsets:
        polygons.extend([(x + ox, y + oy) for x, y in poly] for poly in sub)
    return polygons


def _sierpinski_carpet(depth: int, size: float) -> list[list[tuple[float, float]]]:
    """Generate Sierpinski carpet polygons recursively."""
    hs = size / 2
    if depth == 0:
        return [[(-hs, -hs), (hs, -hs), (hs, hs), (-hs, hs)]]

    polygons: list[list[tuple[float, float]]] = []
    sub_size = size / 3
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue  # skip center
            ox = (i - 1) * sub_size
            oy = (j - 1) * sub_size
            sub = _sierpinski_carpet(depth - 1, sub_size)
            polygons.extend([(x + ox, y + oy) for x, y in poly] for poly in sub)
    return polygons


def _vicsek_cross(depth: int, size: float) -> list[list[tuple[float, float]]]:
    """Generate Vicsek cross/plus fractal polygons."""
    hs = size / 2
    if depth == 0:
        return [[(-hs, -hs), (hs, -hs), (hs, hs), (-hs, hs)]]

    polygons: list[list[tuple[float, float]]] = []
    sub_size = size / 3
    offsets = [
        (0, 0),
        (sub_size, 0),
        (-sub_size, 0),
        (0, sub_size),
        (0, -sub_size),
    ]
    sub = _vicsek_cross(depth - 1, sub_size)
    for ox, oy in offsets:
        polygons.extend([(x + ox, y + oy) for x, y in poly] for poly in sub)
    return polygons


def _vicsek_saltire(depth: int, size: float) -> list[list[tuple[float, float]]]:
    """Generate Vicsek saltire (X-shape) fractal polygons."""
    hs = size / 2
    if depth == 0:
        return [[(-hs, -hs), (hs, -hs), (hs, hs), (-hs, hs)]]

    polygons: list[list[tuple[float, float]]] = []
    sub_size = size / 3
    offsets = [
        (0, 0),
        (sub_size, sub_size),
        (-sub_size, sub_size),
        (sub_size, -sub_size),
        (-sub_size, -sub_size),
    ]
    sub = _vicsek_saltire(depth - 1, sub_size)
    for ox, oy in offsets:
        polygons.extend([(x + ox, y + oy) for x, y in poly] for poly in sub)
    return polygons


@gf.cell_with_module_name(tags=["shapes"])
def fractal(
    fractal_type: Literal[
        "sierpinski_triangle",
        "sierpinski_carpet",
        "vicsek_cross",
        "vicsek_saltire",
    ] = "sierpinski_triangle",
    depth: int = 4,
    size: float = 100.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a fractal pattern.

    Args:
        fractal_type: type of fractal.
        depth: recursion depth (max recommended: 6).
        size: overall size of the fractal.
        layer: layer spec.
    """
    if depth > 6:
        raise ValueError(f"depth={depth} too large, max 6 recommended")

    generators = {
        "sierpinski_triangle": _sierpinski_triangle,
        "sierpinski_carpet": _sierpinski_carpet,
        "vicsek_cross": _vicsek_cross,
        "vicsek_saltire": _vicsek_saltire,
    }

    c = Component()
    polygons = generators[fractal_type](depth, size)
    for poly in polygons:
        c.add_polygon(poly, layer=layer)
    return c


if __name__ == "__main__":
    c = fractal()
    c.show()
