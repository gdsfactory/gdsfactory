from __future__ import annotations

__all__ = ["cutback_component", "cutback_component_mirror"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["pcms"])
def cutback_component(
    component: ComponentSpec = "taper_0p5_to_3_l36",
    cols: int = 4,
    rows: int = 5,
    port1: str = "o1",
    port2: str = "o2",
    bend180: ComponentSpec = "bend_euler180",
    mirror: bool = False,
    mirror1: bool = False,
    mirror2: bool = False,
    straight_length: float | None = None,
    straight_length_pair: float | None = None,
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    radius: float | None = None,
    **kwargs: Any,
) -> Component:
    """Returns a daisy chain of components for measuring their loss.

    Works only for components with 2 ports (input, output).

    Args:
        component: for cutback.
        cols: number of columns.
        rows: number of rows.
        port1: name of first optical port.
        port2: name of second optical port.
        bend180: ubend.
        mirror: Flips component. Useful when 'o2' is the port that you want to route to.
        mirror1: mirrors first component.
        mirror2: mirrors second component.
        straight_length: length of the straight section between cutbacks.
        straight_length_pair: length of the straight section between each component pair.
        cross_section: specification (CrossSection, string or dict).
        straight: straight spec.
        radius: radius for the bends. Defaults to cross_section radius.
        kwargs: component settings.
    """
    return cf.cutback_component(
        component=component,
        cols=cols,
        rows=rows,
        port1=port1,
        port2=port2,
        bend180=bend180,
        mirror=mirror,
        mirror1=mirror1,
        mirror2=mirror2,
        straight_length=straight_length,
        straight_length_pair=straight_length_pair,
        straight=straight,
        cross_section=cross_section,
        radius=radius,
        **kwargs,
    )


cutback_component_mirror = CellAlias(cutback_component, mirror=True)
