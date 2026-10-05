from __future__ import annotations

__all__ = ["taper_parabolic"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.typings import LayerSpec

from .._schematic import taper_schematic


@gf.cell_with_module_name(schematic_function=taper_schematic, tags=["tapers"])
def taper_parabolic(
    length: float = 20,
    width1: float = 0.5,
    width2: float = 5.0,
    exp: float = 0.5,
    npoints: int = 100,
    layer: LayerSpec = "WG",
) -> gf.Component:
    """Returns a parabolic_taper.

    Args:
        length: in um.
        width1: in um.
        width2: in um.
        exp: exponent.
        npoints: number of points.
        layer: layer spec.
    """
    return cf.taper_parabolic(
        length=length,
        width1=width1,
        width2=width2,
        exp=exp,
        npoints=npoints,
        layer=layer,
    )
