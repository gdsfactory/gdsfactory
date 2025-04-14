from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

_list_of_dicts: tuple[dict[str, Any], ...] = (
    dict(length_x=10.0, radius=5.0),
    dict(length_x=20.0, radius=10.0),
)


@gf.cell_with_module_name
def ring_single_array(
    ring: ComponentSpec = "ring_single",
    spacing: float = 5.0,
    list_of_dicts: tuple[dict[str, Any], ...] | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Ring of single bus connected with straights.

    Args:
        ring: ring function.
        spacing: between rings.
        list_of_dicts: settings for each ring.
        cross_section: spec.

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
    ring1 = c << gf.get_component(ring, cross_section=cross_section, **settings0)

    ring0 = ring1
    wg = gf.c.straight(length=spacing, cross_section=cross_section)

    for settings in list_of_dicts[1:]:
        ringi = c << gf.get_component(ring, cross_section=cross_section, **settings)
        wgi = c << wg
        wgi.connect("o1", ring0.ports["o2"])
        ringi.connect("o1", wgi.ports["o2"])
        ring0 = ringi

    c.add_port("o1", port=ring1.ports["o1"])
    c.add_port("o2", port=ringi.ports["o2"])
    return c


if __name__ == "__main__":
    c = ring_single_array(cross_section="strip")
    c.show()
