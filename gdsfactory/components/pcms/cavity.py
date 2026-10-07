from __future__ import annotations

__all__ = ["cavity"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec


@gf.cell_with_module_name(tags=["pcms"])
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

    ```text
      ml (mirror left)              mr (mirror right)
       |                               |
       |o1 - o2__             __o3 - o1|
       |         \           /         |
                  \         /
                ---=========---
         o1  o1    length      o4    o2
    ```

    """
    return cf.cavity(
        component=component,
        coupler=coupler,
        length=length,
        gap=gap,
        **kwargs,
    )
