"""
Ports define where each port has:

- name
- midpoint: (x, y)
- width:
- orientation: (deg) 0, 90, 180, 270.
    where 0 faces east, 90 (north), 180 (west), 270 (south)
"""


from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def test_component_with_port(
    length: int = 5, wg_width: float = 0.5, layer: Tuple[int, int] = gf.LAYER.WG
) -> Component:
    """
    component with one port on the west side
    """

    y = wg_width
    x = length

    c = gf.Component()
    c.add_polygon([(0, 0), (x, 0), (x, y), (0, y)], layer=layer)
    c.add_port(
        name="o1",
        midpoint=(0, y / 2),
        width=y,
        orientation=180,
        layer=layer,
    )
    assert len(c.ports) == 1
    return c


if __name__ == "__main__":
    c = test_component_with_port()
    c.show()
