import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.tech import LAYER
from gdsfactory.types import Layer


@gf.cell
def verniers(
    width_min: float = 0.1,
    width_max: float = 0.5,
    gap: float = 0.1,
    size_max: int = 11,
    layer_label: Layer = LAYER.LABEL,
    **kwargs
) -> Component:
    c = gf.Component()
    y = 0

    widths = np.linspace(width_min, width_max, int(size_max / (width_max + gap)))

    for width in widths:
        w = c << gf.components.straight(width=width, length=size_max, **kwargs)
        y += width / 2
        w.y = y
        c.add_label(text=str(int(width * 1e3)), position=(0, y), layer=layer_label)
        y += width / 2 + gap

    return c


if __name__ == "__main__":
    c = verniers()
    c.show()
