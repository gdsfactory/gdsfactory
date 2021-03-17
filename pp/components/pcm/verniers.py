from typing import Optional

import numpy as np

import pp
from pp.component import Component
from pp.tech import TECH_SILICON_C, Tech


@pp.cell
def verniers(
    width_min: float = 0.1,
    width_max: float = 0.5,
    gap: float = 0.1,
    size_max: int = 11,
    tech: Optional[Tech] = None,
) -> Component:
    c = pp.Component()
    y = 0

    tech = tech or TECH_SILICON_C
    layer_label = tech.layer_label
    widths = np.linspace(width_min, width_max, int(size_max / (width_max + gap)))

    for width in widths:
        w = c << pp.components.waveguide(width=width, length=size_max, tech=tech)
        y += width / 2
        w.y = y
        c.add_label(text=str(int(width * 1e3)), position=(0, y), layer=layer_label)
        y += width / 2 + gap

    return c


if __name__ == "__main__":
    c = verniers()
    c.show()
