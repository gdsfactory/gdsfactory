"""Sagnac loop_mirror."""

from __future__ import annotations

__all__ = ["loop_mirror"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["filters"])
def loop_mirror(
    component: ComponentSpec = "mmi1x2",
    bend90: ComponentSpec = "bend_euler",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns Sagnac loop_mirror.

    Args:
        component: 1x2 splitter.
        bend90: 90 deg bend.
        cross_section: cross_section settings.

    """
    return cf.loop_mirror(
        component=component, bend90=bend90, cross_section=cross_section
    )
