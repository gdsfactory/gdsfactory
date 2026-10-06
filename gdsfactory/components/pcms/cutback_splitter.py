from __future__ import annotations

__all__ = ["cutback_splitter"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["pcms"])
def cutback_splitter(
    component: ComponentSpec = "mmi1x2",
    cols: int = 4,
    rows: int = 5,
    port1: str = "o1",
    port2: str = "o2",
    port3: str = "o3",
    bend180: ComponentSpec = "bend_euler180",
    mirror: bool = False,
    straight: ComponentSpec = "straight",
    straight_length: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    **kwargs: Any,
) -> Component:
    """Returns a daisy chain of splitters for measuring their loss.

    Args:
        component: for cutback.
        cols: number of columns.
        rows: number of rows.
        port1: name of first optical port.
        port2: name of second optical port.
        port3: name of third optical port.
        bend180: ubend.
        mirror: Flips component. Useful when 'o2' is the port that you want to route to.
        straight: waveguide spec to connect both sides.
        straight_length: length of the straight section between cutbacks.
        cross_section: specification (CrossSection, string or dict).
        kwargs: cross_section settings.
    """
    return cf.cutback_splitter(
        component=component,
        cols=cols,
        rows=rows,
        port1=port1,
        port2=port2,
        port3=port3,
        bend180=bend180,
        mirror=mirror,
        straight=straight,
        straight_length=straight_length,
        cross_section=cross_section,
        **kwargs,
    )
