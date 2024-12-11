from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec


@gf.cell
def cavity(
    component: ComponentSpec = "dbr",
    coupler: ComponentSpec = "coupler",
    length: float = 0.1,
    gap: float = 0.2,
    **kwargs: Any,
) -> Component:
    r"""Returns  cavity from a coupler and a mirror.

    connects the W0 port of the mirror to E1 and W1 coupler ports
    creating a resonant cavity

    Args:
        component: mirror.
        coupler: coupler library.
        length: coupler length.
        gap: coupler gap.
        kwargs: coupler_settings.

    .. code::

      ml (mirror left)              mr (mirror right)
       |                               |
       |o1 - o2__             __o3 - o1|
       |         \           /         |
                  \         /
                ---=========---
         o1  o1    length      o4    o2

    """
    mirror = gf.get_component(component)
    coupler = gf.get_component(coupler, length=length, gap=gap, **kwargs)

    c = gf.Component()
    cr = c << coupler
    ml = c << mirror
    mr = c << mirror

    ml.connect("o1", other=cr.ports["o2"])
    mr.connect("o1", other=cr.ports["o3"])
    c.add_port("o1", port=cr.ports["o1"])
    c.add_port("o2", port=cr.ports["o4"])
    c.copy_child_info(mirror)
    return c


if __name__ == "__main__":
    from gdsfactory.components import dbr

    c = cavity(component=dbr())
    c.show()
