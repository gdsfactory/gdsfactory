from __future__ import annotations

__all__ = ["bendu_double", "cutback_2x2", "straight_double"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["pcms"])
def bendu_double(
    component: Component,
    cross_section: CrossSectionSpec = "strip",
    bend180: ComponentSpec = "bend_circular180",
    port1: str = "o1",
    port2: str = "o2",
) -> Component:
    """Returns double bend.

    Args:
        component: for cutback.
        cross_section: specification (CrossSection, string or dict).
        bend180: ubend.
        port1: name of first optical port.
        port2: name of second optical port.
    """
    return cf.bendu_double(
        component=component,
        cross_section=cross_section,
        bend180=bend180,
        port1=port1,
        port2=port2,
    )


@gf.cell_with_module_name(tags=["pcms"])
def straight_double(
    component: Component,
    cross_section: CrossSectionSpec = "strip",
    port1: str = "o1",
    port2: str = "o2",
    straight_length: float | None = None,
    straight: ComponentSpec = "straight",
) -> Component:
    """Returns double straight.

    Args:
        component: for cutback.
        cross_section: specification (CrossSection, string or dict).
        port1: name of first optical port.
        port2: name of second optical port.
        straight_length: length of straight.
        straight: straight spec.
    """
    return cf.straight_double(
        component=component,
        cross_section=cross_section,
        port1=port1,
        port2=port2,
        straight_length=straight_length,
        straight=straight,
    )


@gf.cell_with_module_name(tags=["pcms"])
def cutback_2x2(
    component: ComponentSpec = "mmi2x2",
    cols: int = 4,
    rows: int = 5,
    port1: str = "o1",
    port2: str = "o2",
    port3: str = "o3",
    port4: str = "o4",
    bend180: ComponentSpec = "bend_circular180",
    mirror: bool = False,
    straight_length: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    straight: ComponentSpec = "straight",
) -> Component:
    """Returns a daisy chain of splitters for measuring their loss.

    Args:
        component: for cutback.
        cols: number of columns.
        rows: number of rows.
        port1: name of first optical port.
        port2: name of second optical port.
        port3: name of third optical port.
        port4: name of fourth optical port.
        bend180: ubend.
        mirror: Flips component. Useful when 'o2' is the port that you want to route to.
        straight_length: length of the straight section between cutbacks.
        cross_section: specification (CrossSection, string or dict).
        straight: straight spec.
    """
    return cf.cutback_2x2(
        component=component,
        cols=cols,
        rows=rows,
        port1=port1,
        port2=port2,
        port3=port3,
        port4=port4,
        bend180=bend180,
        mirror=mirror,
        straight_length=straight_length,
        cross_section=cross_section,
        straight=straight,
    )
