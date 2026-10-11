"""Hecken taper for microstrip impedance matching.

Adapted from PHIDL <https://github.com/amccaugh/phidl/> by Adam McCaughan
"""

from __future__ import annotations

__all__ = ["taper_hecken"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec

from .._schematic import taper_schematic


@gf.cell_with_module_name(schematic_function=taper_schematic, tags=["tapers"])
def taper_hecken(
    length: float = 200,
    B: float = 4.0091,
    dielectric_thickness: float = 0.25,
    eps_r: float = 2,
    Lk_per_sq: float = 250e-12,
    Z1: float | None = 50,
    Z2: float | None = 100,
    width1: float | None = None,
    width2: float | None = None,
    num_pts: int = 100,
    layer: LayerSpec = "WG",
) -> Component:
    """Creates a Hecken-tapered microstrip.

    Args:
        length: Length of the microstrip.
        B: Controls the intensity of the taper.
        dielectric_thickness: Thickness of the substrate.
        eps_r: Dielectric constant of the substrate.
        Lk_per_sq: Kinetic inductance per square of the microstrip.
        Z1: Impedance of the left side region of the microstrip.
        Z2: Impedance of the right side region of the microstrip.
        width1: Width of the left side of the microstrip.
        width2: Width of the right side of the microstrip.
        num_pts: Number of points comprising the curve of the entire microstrip.
        layer: Specific layer(s) to put polygon geometry on.

    Returns:
        Component containing a Hecken-tapered microstrip.
    """
    return cf.taper_hecken(
        length=length,
        B=B,
        dielectric_thickness=dielectric_thickness,
        eps_r=eps_r,
        Lk_per_sq=Lk_per_sq,
        Z1=Z1,
        Z2=Z2,
        width1=width1,
        width2=width2,
        num_pts=num_pts,
        layer=layer,
    )


if __name__ == "__main__":
    c = taper_hecken(Z1=50, Z2=100)
    c.show()
