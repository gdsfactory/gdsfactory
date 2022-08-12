from typing import Dict, Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.ring_single import ring_single
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.types import ComponentFactory

_list_of_dicts = (
    dict(length_x=10.0, radius=5.0),
    dict(length_x=20.0, radius=10.0),
)


@gf.cell
def ring_single_array(
    ring: ComponentFactory = ring_single,
    straight: ComponentFactory = straight_function,
    spacing: float = 5.0,
    list_of_dicts: Optional[Tuple[Dict[str, float], ...]] = None,
) -> Component:
    """Ring of single bus connected with straights.

    Args:
        ring: ring function.
        straight: straight function.
        spacing: between rings.
        list_of_dicts: settings for each ring.

    .. code::

           ______               ______
          |      |             |      |
          |      |  length_y   |      |
          |      |             |      |
         --======-- spacing ----==gap==--

          length_x
    """
    list_of_dicts = list_of_dicts or _list_of_dicts
    c = Component()
    settings0 = list_of_dicts[0]
    ring1 = c << ring(**settings0)

    ring0 = ring1
    wg = straight(length=spacing)

    for settings in list_of_dicts[1:]:
        ringi = c << ring(**settings)
        wgi = c << wg
        wgi.connect("o1", ring0.ports["o2"])
        ringi.connect("o1", wgi.ports["o2"])
        ring0 = ringi

    c.add_port("o1", port=ring1.ports["o1"])
    c.add_port("o2", port=ringi.ports["o2"])
    return c


if __name__ == "__main__":
    c = ring_single_array()
    c.show(show_ports=True)
