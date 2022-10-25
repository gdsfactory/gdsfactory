"""Ports define where each port has the follow properties.

- name
- center: (x, y)
- width:
- orientation: (deg) 0, 90, 180, 270.
    where 0 faces east, 90 (north), 180 (west), 270 (south)

"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.types import LayerSpec


@gf.cell
def test_component_with_port(
    length: float = 5.0, width: float = 0.5, layer: LayerSpec = "WG"
) -> Component:
    """Returns a component with one port on the west side.

    Args:
        length: in um.
        width: waveguide width in um.
        layer: layer.

    """
    y = width
    x = length

    c = gf.Component()
    c.add_polygon([(0, 0), (x, 0), (x, y), (0, y)], layer=layer)
    c.add_port(name="o1", center=(0, y / 2), width=y, orientation=180, layer=layer)
    assert len(c.ports) == 1
    return c


if __name__ == "__main__":
    c = test_component_with_port()
    c.show(show_ports=True)
