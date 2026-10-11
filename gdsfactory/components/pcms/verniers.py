from __future__ import annotations

__all__ = ["verniers"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats, LayerSpec


@gf.cell_with_module_name(tags=["pcms"])
def verniers(
    widths: Floats = (0.1, 0.2, 0.3, 0.4, 0.5),
    gap: float = 0.1,
    xsize: float = 100.0,
    layer_label: LayerSpec = "TEXT",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    **kwargs: Any,
) -> Component:
    """Returns a component with verniers.

    Args:
        widths: list of widths.
        gap: gap between verniers.
        xsize: size of the component.
        layer_label: layer for the labels.
        straight: straight function.
        cross_section: cross_section spec.
        kwargs: straight settings.
    """
    return cf.verniers(
        widths=widths,
        gap=gap,
        xsize=xsize,
        layer_label=layer_label,
        straight=straight,
        cross_section=cross_section,
        **kwargs,
    )
