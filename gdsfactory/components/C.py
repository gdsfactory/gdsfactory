from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
def C(
    width: float = 1.0,
    size: tuple[float, float] = (10.0, 20.0),
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """C geometry with ports on both ends.

    based on phidl.

    Args:
        width: of the line.
        size: length and height of the base.
        layer: layer spec.
        port_type: optical or electrical.

    .. code::

         ______
        |       o1
        |   ___
        |  |
        |  |___
        ||<---> size[0]
        |______ o2
    """
    layer = gf.get_layer(layer)
    c = Component()
    w = width / 2
    s1, s2 = size
    points = [
        (-w, -w),
        (s1, -w),
        (s1, w),
        (w, w),
        (w, s2 - w),
        (s1, s2 - w),
        (s1, s2 + w),
        (-w, s2 + w),
        (-w, -w),
    ]
    c.add_polygon(points, layer=layer)
    port_settings = dict(width=width, orientation=0, layer=layer, port_type=port_type)
    c.add_port(name="o1", center=(s1, s2), **port_settings)
    c.add_port(name="o2", center=(s1, 0), **port_settings)
    return c


if __name__ == "__main__":
    c = C(width=1.0)
    c.show()
