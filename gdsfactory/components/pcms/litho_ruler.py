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
    for n in range(num_marks):
        h = height * scale[n % len(scale)]
        ref = D << gf.components.rectangle(size=(width, h), layer=layer)
        ref.movex((n - num_marks / 2) * pitch + spacing / 2.0)

    return D


@gf.cell_with_module_name
def litho_ruler_staircase(
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
    long_marks: tuple[float, ...] = (-50, 0),
    text_size: float = 3.5,
) -> gf.Component:
    """Ruler structure for lithographic measurement.

    Includes marks of varying scales to allow for easy reading by eye.

    Args:
        height_long: Height of the long ruling marks in um.
        width: Width of the ruling marks in um.
        spacing: Center-to-center spacing of the ruling marks in um.
        marks: list of marks.
        layer: Specific layer to put the ruler geometry on.
    """
    ymin = 0
    c = gf.Component()
    for i, mark in enumerate(marks):
        h = height_numbered if mark else height_short
        h = height_long if mark in long_marks else h

        if mark in long_marks:
            ymin = 0
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

    return c


if __name__ == "__main__":
    c = litho_ruler_staircase()
    c.show()
