from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats, LayerSpec


@gf.cell_with_module_name
def verniers(
    widths: Floats = (0.1, 0.2, 0.3, 0.4, 0.5),
    gap: float = 0.1,
    xsize: float = 100.0,
    layer_label: LayerSpec = "TEXT",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip_no_ports",
    **kwargs: Any,
) -> Component:
    """Returns a component with verniers.

    Args:
        widths: list of widths.
        gap: gap between verniers.
        xsize: size of the component.
        layer_label: layer for the labels.
        straight: straight function.
        cross_section: cross_section spec.
        kwargs: straight settings.
    """
    c = gf.Component()
    y = 0.0

    for width in widths:
        w = c << gf.get_component(
            straight, width=width, length=xsize, cross_section=cross_section, **kwargs
        )
        y += width / 2
        w.y = y
        c.add_label(text=str(int(width * 1e3)), position=(0, y), layer=layer_label)
        y += width / 2 + gap

    return c


if __name__ == "__main__":
    c = verniers()
    c.show()
