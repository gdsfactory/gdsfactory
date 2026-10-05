from __future__ import annotations

__all__ = ["bbox", "bbox_to_points"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import ComponentReference
from gdsfactory.component_functions.shapes.bbox import bbox_to_points
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def bbox(
    component: gf.Component | ComponentReference,
    layer: LayerSpec,
    top: float = 0,
    bottom: float = 0,
    left: float = 0,
    right: float = 0,
) -> gf.Component:
    """Returns bounding box rectangle from coordinates.

    Args:
        component: component or instance to get bbox from.
        layer: for bbox.
        top: north offset.
        bottom: south offset.
        left: west offset.
        right: east offset.
    """
    return cf.bbox(
        component=component, layer=layer, top=top, bottom=bottom, left=left, right=right
    )
