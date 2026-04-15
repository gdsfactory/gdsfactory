from __future__ import annotations

__all__ = ["rect_su_shape"]

import gdsfactory as gf
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
    c = Component()
    w = width / 2

    # Build as three connected rectangles
    # Segment 1: vertical from (0,0) going up by L1
    # Segment 2: horizontal at top of seg1, going right by L2
    # Segment 3: vertical from end of seg2, going up by L3
    # The overall shape is an S/U polygon

    if L1 >= 0 and L2 >= 0 and L3 >= 0:
        points = [
            (-w, 0),
            (w, 0),
            (w, L1),
            (L2 + w, L1),
            (L2 + w, L1 + L3),
            (L2 - w, L1 + L3),
            (L2 - w, L1 + width),
            (-w, L1 + width),
            (-w, 0),
        ]
    else:
        # General case: compose three rectangles
        r1 = gf.components.rectangle(size=(width, abs(L1)), layer=layer)
        r2 = gf.components.rectangle(size=(abs(L2), width), layer=layer)
        r3 = gf.components.rectangle(size=(width, abs(L3)), layer=layer)

        ref1 = c.add_ref(r1)
        ref1.center = (0, 0)

        ref2 = c.add_ref(r2)
        if L1 >= 0:
            ref2.xmin = -w
            ref2.ymin = abs(L1) / 2 - w
        else:
            ref2.xmin = -w
            ref2.ymax = -abs(L1) / 2 + w

        ref3 = c.add_ref(r3)
        if L2 >= 0:
            ref3.xmin = abs(L2) - w
        else:
            ref3.xmax = -abs(L2) + w
        if L3 >= 0:
            ref3.ymin = ref2.ymin
        else:
            ref3.ymax = ref2.ymax

        c.flatten()
        return c

    c.add_polygon(points, layer=layer)

    if port_type:
        prefix = "o" if port_type == "optical" else "e"
        c.add_port(
            f"{prefix}1",
            center=(0, 0),
            width=width,
            orientation=270,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}2",
            center=(L2, L1 + L3),
            width=width,
            orientation=90,
            layer=layer,
            port_type=port_type,
        )
        c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = rect_su_shape()
    c.show()
