"""You can add ports to connect components ."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import LayerSpec


@gf.cell
def straight_narrow(
    length: float = 5.0, width: float = 0.3, layer: LayerSpec = (1, 0)
) -> gf.Component:
    """Returns straight Component.

    Args:
        length: of the straight.
        width: in um.
        layer: layer spec.
    """
    wg = gf.Component("straight_sample")
    wg.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    wg.add_port(
        name="o1", center=(0, width / 2), width=width, orientation=180, layer=layer
    )
    wg.add_port(
        name="o2", center=(length, width / 2), width=width, orientation=0, layer=layer
    )
    return wg


if __name__ == "__main__":
    c = straight_narrow()
    c.show()
