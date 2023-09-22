from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import Floats, LayerSpec


@gf.cell
def verniers(
    widths: Floats = (0.1, 0.2, 0.3, 0.4, 0.5),
    gap: float = 0.1,
    xsize: float = 100.0,
    layer_label: LayerSpec = "TEXT",
    **kwargs,
) -> Component:
    """Returns a component with verniers.

    Args:
        widths: list of widths.
        gap: gap between verniers.
        xsize: size of the component.
        layer_label: layer for the labels.
        **kwargs: cross_section settings.
    """
    c = gf.Component()
    y = 0

    for width in widths:
        w = c << gf.components.straight(width=width, length=xsize, **kwargs)
        y += width / 2
        w.y = y
        c.add_label(text=str(int(width * 1e3)), position=(0, y), layer=layer_label)
        y += width / 2 + gap

    return c


if __name__ == "__main__":
    c = verniers()
    c.show(show_ports=True)
