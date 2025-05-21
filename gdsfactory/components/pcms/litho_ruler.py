from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def litho_ruler(
    height: float = 2,
    width: float = 0.5,
    spacing: float = 2.0,
    scale: tuple[float, ...] = (3, 1, 1, 1, 1, 2, 1, 1, 1, 1),
    num_marks: int = 21,
    layer: LayerSpec = "WG",
) -> gf.Component:
    """Ruler structure for lithographic measurement.

    Includes marks of varying scales to allow for easy reading by eye.

    based on phidl.geometry

    Args:
        height: Height of the ruling marks in um.
        width: Width of the ruling marks in um.
        spacing: Center-to-center spacing of the ruling marks in um.
        scale: Height scale pattern of marks.
        num_marks: Total number of marks to generate.
        layer: Specific layer to put the ruler geometry on.
    """
    pitch = spacing + width
    D = gf.Component()
    # Cache rectangles by their effective height
    rect_cache = {}
    scale_list = scale  # faster local lookup
    n_scale = len(scale)
    w = width
    h0 = height
    ly = layer
    for n in range(num_marks):
        s = scale_list[n % n_scale]
        h = h0 * s
        key = (w, h, ly)
        if key not in rect_cache:
            rect_cache[key] = gf.components.rectangle(size=(w, h), layer=ly)
        ref = D << rect_cache[key]
        ref.movex((n - num_marks / 2) * pitch)
    return D


if __name__ == "__main__":
    c = litho_ruler()
    c.show()
