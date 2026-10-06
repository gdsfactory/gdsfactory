from __future__ import annotations

__all__ = ["ring"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["rings"])
def ring(
    radius: float = 10.0,
    width: float = 0.5,
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
    angle: float = 360,
    distance_resolution: float | None = None,
) -> Component:
    """Returns a ring.

    Args:
        radius: ring radius.
        width: of the ring.
        angle_resolution: max number of degrees per point.
        layer: layer.
        angle: angular coverage of the ring
        distance_resolution: max distance between points. This is an alternate way to describe the resolution besides setting angle_resolution. If distance_resolution and angle_resolution are both set, distance_resolution determines the resolution.
    """
    return cf.ring(
        radius=radius,
        width=width,
        angle_resolution=angle_resolution,
        layer=layer,
        angle=angle,
        distance_resolution=distance_resolution,
    )
