from __future__ import annotations

__all__ = ["fractal"]

from typing import Literal

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


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
    return cf.fractal(fractal_type=fractal_type, depth=depth, size=size, layer=layer)


if __name__ == "__main__":
    c = fractal()
    c.show()
