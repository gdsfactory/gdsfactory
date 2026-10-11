from __future__ import annotations

__all__ = ["ring_single_array"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["rings"])
def ring_single_array(
    ring: ComponentSpec = "ring_single",
    spacing: float = 15.0,
    list_of_dicts: tuple[dict[str, Any], ...] | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Ring of single bus connected with straights.

    Args:
        ring: ring spec.
        spacing: between rings.
        list_of_dicts: settings for each ring.
        cross_section: spec.

    ```text
           ______               ______
          |      |             |      |
          |      |  length_y   |      |
          |      |             |      |
         --======-- spacing ----==gap==--
    ```

          length_x
    """
    return cf.ring_single_array(
        ring=ring,
        spacing=spacing,
        list_of_dicts=list_of_dicts,
        cross_section=cross_section,
    )
