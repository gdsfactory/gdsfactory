"""Ports define where each port has the follow properties.

- name
- center: (x, y)
- width:
- orientation: (deg) 0, 90, 180, 270.
    where 0 faces east, 90 (north), 180 (west), 270 (south)

"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
def component_with_port(
    length: float = 5.0, width: float = 0.5, layer: LayerSpec = "WG"
) -> Component:
    """Returns a component with one port on the west side.

    Args:
        length: in um.
        width: waveguide width in um.
        layer: layer.
    """
    c = gf.Component()
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    c.add_port(
        name="o1", center=(0, width / 2), width=width, orientation=180, layer=layer
    )
    assert len(c.ports) == 1
    return c


def test_component_with_port() -> None:
    assert component_with_port()


if __name__ == "__main__":
    c = component_with_port()
    c.show()
