from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
def hline(
    length: float = 10.0,
    width: float = 0.5,
    layer: LayerSpec = "WG",
    port_type: str = "optical",
) -> Component:
    """Horizontal line straight, with ports on east and west sides."""
    c = gf.Component()
    if length > 0 and width > 0:
        a = width / 2
        c.add_polygon([(0, -a), (length, -a), (length, a), (0, a)], layer=layer)

    c.add_port(
        name="o1",
        center=(0.0, 0.0),
        width=width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        name="o2",
        center=(length, 0.0),
        width=width,
        orientation=0,
        layer=layer,
        port_type=port_type,
    )

    c.info["width"] = width
    c.info["length"] = length
    return c


if __name__ == "__main__":
    c = hline(width=10)
    print(c)
    c.show(show_ports=True)
