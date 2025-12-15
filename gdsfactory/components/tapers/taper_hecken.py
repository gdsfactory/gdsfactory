"""Hecken taper for microstrip impedance matching.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

__all__ = ["taper_hecken"]

import numpy as np
from numpy import log

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec

from ..analog.microstrip import (
    _G,
    _find_microstrip_wire_width,
    _microstrip_v_with_Lk,
    _microstrip_Z_with_Lk,
)


@gf.cell_with_module_name
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
    if width1 is not None:
        Z1 = _microstrip_Z_with_Lk(
            width1 * 1e-6, dielectric_thickness * 1e-6, eps_r, Lk_per_sq
        )
    if width2 is not None:
        Z2 = _microstrip_Z_with_Lk(
            width2 * 1e-6, dielectric_thickness * 1e-6, eps_r, Lk_per_sq
        )
    # Normalized length of the wire [-1 to +1]
    xi_list = np.linspace(-1, 1, num_pts)
    if Z1 is None or Z2 is None:
        raise ValueError(
            "Z1 and Z2 must be specified either directly or via width1/width2"
        )
    Z = [np.exp(0.5 * log(Z1 * Z2) + 0.5 * log(Z2 / Z1) * _G(xi, B)) for xi in xi_list]
    widths = np.array(
        [
            _find_microstrip_wire_width(
                z, dielectric_thickness * 1e-6, eps_r, Lk_per_sq
            )
            * 1e6
            for z in Z
        ]
    )
    x = (xi_list / 2) * length

    # Compensate for varying speed of light in the microstrip by shortening
    # and lengthening sections according to the speed of light in that section
    v = np.array(
        [
            _microstrip_v_with_Lk(
                w * 1e-6, dielectric_thickness * 1e-6, eps_r, Lk_per_sq
            )
            for w in widths
        ]
    )
    dx = np.diff(x)
    dx_compensated = dx * v[:-1]
    x_compensated = np.cumsum(dx_compensated)
    x = np.hstack([0, x_compensated]) / max(x_compensated) * length

    # Create blank device and add taper polygon
    c = gf.Component()
    xpts = np.concatenate([x, x[::-1]])
    ypts = np.concatenate([widths / 2, -widths[::-1] / 2])
    points = np.column_stack((xpts, ypts))
    c.add_polygon(points, layer=layer)
    # Snap port widths to even multiples of 0.002 um (2 dbu)
    width1_snapped = round(widths[0] / 0.002) * 0.002
    width2_snapped = round(widths[-1] / 0.002) * 0.002
    c.add_port(
        name="o1", center=(0, 0), width=width1_snapped, orientation=180, layer=layer
    )
    c.add_port(
        name="o2", center=(length, 0), width=width2_snapped, orientation=0, layer=layer
    )

    # Add meta information about the taper
    c.info["num_squares"] = float(np.sum(np.diff(x) / widths[:-1]))
    c.info["width1"] = float(widths[0])
    c.info["width2"] = float(widths[-1])
    c.info["Z1"] = float(Z[0])
    c.info["Z2"] = float(Z[-1])
    # Note there are two values for v/c (and f_cutoff) because the speed of
    # light is different at the beginning and end of the taper

    # c.info["w"] = widths.tolist()
    # c.info["x"] = x.tolist()
    # c.info["Z"] = Z if isinstance(Z, list) else list(Z)
    # c.info["v/c"] = (v / 3e8).tolist()

    time_length = float(np.sum(np.diff(x * 1e-6) / (v[:-1])))
    c.info["time_length"] = time_length
    c.info["f_cutoff"] = 1 / (2 * time_length)
    c.info["length"] = float(length)
    return c


if __name__ == "__main__":
    c = taper_hecken(Z1=50, Z2=100)
    c.show()
