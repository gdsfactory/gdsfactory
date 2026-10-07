from __future__ import annotations

__all__ = ["via_corner"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.cross_section import metal2, metal3
from gdsfactory.typings import ComponentSpec, MultiCrossSectionAngleSpec


@gf.cell_with_module_name(tags=["vias"])
def via_corner(
    cross_section: MultiCrossSectionAngleSpec = (
        (metal2, (0, 180)),
        (metal3, (90, 270)),
    ),
    vias: tuple[ComponentSpec] = ("via1",),
    layers_labels: tuple[str, ...] = ("m2", "m3"),
    **kwargs: Any,
) -> gf.Component:
    """Returns Corner via.

    Use in place of wire_corner to route between two layers.

    Args:
        cross_section: list of cross_section, orientation pairs.
        vias: vias to use to fill the rectangles.
        layers_labels: Labels to use for each layer.
        kwargs: cross_section settings.
    """
    return cf.via_corner(
        cross_section=cross_section,
        vias=vias,
        layers_labels=layers_labels,
        **kwargs,
    )
