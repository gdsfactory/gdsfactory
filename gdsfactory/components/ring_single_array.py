from typing import Dict, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.ring_single import ring_single
from gdsfactory.components.straight import straight
from gdsfactory.types import ComponentFactory


@gf.cell
def ring_single_array(
    ring_function: ComponentFactory = ring_single,
    straight_function: ComponentFactory = straight,
    spacing: float = 5.0,
    list_of_dicts: Tuple[Dict[str, float], ...] = (
        dict(length_x=10.0, radius=5.0),
        dict(length_x=20.0, radius=10.0),
    ),
) -> Component:
    """Ring of single bus connected with straights.

    .. code::

           ______               ______
          |      |             |      |
          |      |  length_y   |      |
          |      |             |      |
         --======-- spacing ----==gap==--

          length_x

    """
    c = Component()
    settings0 = list_of_dicts[0]
    ring1 = c << ring_function(**settings0)

    ring0 = ring1
    wg = straight_function(length=spacing)

    for settings in list_of_dicts[1:]:
        ringi = c << ring_function(**settings)
        wgi = c << wg
        wgi.connect(1, ring0.ports[2])
        ringi.connect(1, wgi.ports[2])
        ring0 = ringi

    c.add_port(1, port=ring1.ports[1])
    c.add_port(2, port=ringi.ports[2])
    return c


if __name__ == "__main__":
    c = ring_single_array()
    c.show()
