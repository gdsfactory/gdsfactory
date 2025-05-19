from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from numpy import sin, sqrt

import gdsfactory as gf
from gdsfactory.functions import DEG2RAD, extrude_path

neff_ridge = 2.8
neff_shallow = 2.5


def ellipse_arc(
    a: float,
    b: float,
    x0: float,
    theta_min: float,
    theta_max: float,
    angle_step: float = 0.5,
) -> npt.NDArray[np.floating[Any]]:
    """Returns an elliptical arc.

    b = a *sqrt(1-e**2)

    An ellipse with a = b has zero eccentricity (is a circle)

    Args:
        a: ellipse semi-major axis.
        b: semi-minor axis.
        x0: in um.
        theta_min: in rad.
        theta_max: in rad.
        angle_step: in rad.
    """
    # Compute theta in degrees, then convert to radians only once
    theta = np.arange(theta_min, theta_max + angle_step, angle_step) * DEG2RAD

    # Compute xs and ys in a vectorized way, no unnecessary temporaries
    xs = a * np.cos(theta) + x0
    ys = b * np.sin(theta)
    # Stack arrays first, then snap both columns at once for better cache locality and fewer function calls
    arc = np.empty((xs.size, 2), dtype=float)
    arc[:, 0] = xs
    arc[:, 1] = ys
    arc = gf.snap.snap_to_grid(arc)
    return arc  # shape (N,2), as before


def grating_tooth_points(
    ap: float,
    bp: float,
    xp: float,
    width: float,
    taper_angle: float,
    spiked: bool = True,
    angle_step: float = 1.0,
) -> npt.NDArray[np.floating[Any]]:
    theta_min = -taper_angle / 2
    theta_max = taper_angle / 2

    backbone_points = ellipse_arc(ap, bp, xp, theta_min, theta_max, angle_step)
    spike_length = width / 3 if spiked else 0.0
    return extrude_path(
        backbone_points,
        width,
        with_manhattan_facing_angles=False,
        spike_length=spike_length,
    )


def grating_taper_points(
    a: float,
    b: float,
    x0: float,
    taper_length: float,
    taper_angle: float,
    wg_width: float,
    angle_step: float = 1.0,
) -> npt.NDArray[np.floating[Any]]:
    """Returns a taper for a grating coupler.

    Args:
        a: ellipse semi-major axis.
        b: semi-minor axis.
        x0: in um.
        taper_length: in um.
        taper_angle: in degrees.
        wg_width: in um.
        angle_step: in degrees.
    """
    # ellipse_arc is already optimized and returns snapped arc
    taper_arc = ellipse_arc(
        a=a,
        b=b,
        x0=taper_length,
        theta_min=-taper_angle / 2,
        theta_max=taper_angle / 2,
        angle_step=angle_step,
    )

    # Use np.array for all at once, no need for multiple arrays + addition
    p0 = np.array([x0, wg_width / 2], dtype=float)
    p1 = np.array([x0, -wg_width / 2], dtype=float)
    # Allocate pre-sized array for output, which minimizes memory reallocations
    out = np.empty((taper_arc.shape[0] + 2, 2), dtype=float)
    out[0] = p0
    out[1] = p1
    out[2:] = taper_arc
    return out


def get_grating_period_curved(
    fiber_angle: float = 15.0,
    wavelength: float = 1.55,
    n_slab: float = (neff_ridge + neff_shallow) / 2,
    n_clad: float = 1.0,
) -> tuple[float, float]:
    """The following function calculates the confocal grating periods n_slab is.

    the "average slab index" of the grating. For 220nm silicon it is 2.8, for
    150nm it is 2.5. The average is approximately 2.65. n_clad is the cladding
    index in which the fiber is located, not the index of the layer above the
    straight. If the fiber is in air, then it is 1.0. If you use an index
    matching fluid or glue, then it should be 1.45.

    Args:
        fiber_angle: in degrees.
        wavelength: um.
        n_slab: slab refractive index.
        n_clad: cladding refractive index.
    """
    sin_fiber_angle = sin(DEG2RAD * fiber_angle)
    n_clad_cos = n_clad * sin_fiber_angle
    n2_slab = n_slab * n_slab
    n2_clad_cos2 = n_clad_cos * n_clad_cos
    n2_reduced = n2_slab - n2_clad_cos2
    sqrt_n2_reduced = sqrt(n2_reduced)
    # The following arithmetic preserves the formulae as before, but uses the precalculated variables.
    h_period = wavelength * (n_slab + n_clad_cos) / n2_reduced
    v_period = wavelength / sqrt_n2_reduced
    return h_period, v_period


def get_grating_period(
    fiber_angle: float = 13.45,
    wavelength: float = 1.55,
    neff_high: float = neff_ridge,
    neff_low: float = neff_shallow,
    n_clad: float = 1.45,
) -> float:
    """Return grating coupler period based on lumerical slides.

    Args:
        fiber_angle: in degrees.
        wavelength: um.
        neff_high: high index.
        neff_low: low index.
        n_clad: cladding index.
    """
    neff = (neff_high + neff_low) / 2
    return wavelength / (neff - float(sin(DEG2RAD * fiber_angle)) * n_clad)


if __name__ == "__main__":
    pc = get_grating_period_curved()
    p = get_grating_period()
    print(pc)
    print(p)
