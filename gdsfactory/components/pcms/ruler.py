from __future__ import annotations

__all__ = ["ruler"]

import gdsfactory as gf
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def ruler(
    height_long: float = 55,
    height_short: float = 5,
    height_numbered: float = 10,
    width: float = 2,
    spacing: float = 5.0,
    marks: tuple[float | None, ...] = (
        -100,
        None,
        -90,
        None,
        -80,
        None,
        -70,
        None,
        -60,
        None,
        -50,
        None,
        -40,
        None,
        -30,
        None,
        -20,
        None,
        -10,
        None,
        0,
    ),
    layer: LayerSpec = "WG",
    bbox_layers: tuple[LayerSpec, ...] | None = None,
    bbox_offset: float = 3.0,
    long_marks: tuple[float, ...] = (-50, 0),
    text_size: float = 3.5,
) -> gf.Component:
    """Ruler structure for lithographic measurement.

    Includes marks of varying scales to allow for easy reading by eye.

    Args:
        height_long: Height of the long ruling marks in um.
        height_short: Height of the short ruling marks in um.
        height_numbered: Height of the numbered ruling marks in um.
        width: Width of the ruling marks in um.
        spacing: Center-to-center spacing of the ruling marks in um.
        marks: Height scale pattern of marks.
        layer: Specific layer to put the ruler geometry on.
        bbox_layers: Layers to include in the bounding box.
        bbox_offset: Offsets for each bounding box layer.
        cross_section: Cross-section spec for the ruler. Overrides layer if provided.
        long_marks: Marks that are long.
        text_size: Size of the text in um.
    """
    ymin = 0.0
    c = gf.Component()
    for i, mark in enumerate(marks):
        h = height_numbered if mark else height_short
        h = height_long if mark in long_marks else h

        if mark in long_marks:
            ymin = 0.0
        else:
            ymin += height_short

        ref = c << gf.components.rectangle(size=(width, h), layer=layer)
        ref.xmin = i * spacing
        ref.ymin = ymin

        if mark is not None:
            t = c << gf.c.text_rectangular(
                text=str(mark), size=text_size / 5, layer=layer
            )
            t.rotate(90)
            t.ymin = ref.ymin + 1
            t.xmax = ref.xmin - 1
    if bbox_layers:
        gf.add_padding(c, layers=bbox_layers, default=bbox_offset)
    return c


if __name__ == "__main__":
    c = ruler(bbox_layers=("SLAB90",), bbox_offset=2)
    c.show()
