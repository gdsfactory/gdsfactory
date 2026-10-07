from __future__ import annotations

__all__ = ["rect_su_shape"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def rect_su_shape(
    L1: float = 10.0,
    L2: float = 10.0,
    L3: float = 20.0,
    width: float = 1.0,
    layer: LayerSpec = "WG",
    port_type: str | None = "electrical",
) -> Component:
    """Returns a rectangular S- or U-shaped routing structure.

    The shape consists of three connected rectangular segments forming
    an S or U pattern. Positive and negative values of L1, L2, L3
    produce different orientations (S-shape, U-shape, etc.).

    Args:
        L1: length of first vertical segment.
        L2: length of horizontal segment.
        L3: length of second vertical segment.
        width: width of all segments.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    return cf.rect_su_shape(
        L1=L1, L2=L2, L3=L3, width=width, layer=layer, port_type=port_type
    )


if __name__ == "__main__":
    c = rect_su_shape()
    c.show()
