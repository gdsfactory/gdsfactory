from __future__ import annotations

from numpy import pi, sin, sqrt

neff_ridge = 2.8
neff_shallow = 2.5


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
    DEG2RAD = pi / 180
    cos_fib_angle = sin(DEG2RAD * fiber_angle)
    n2_reduced = n_slab**2 - n_clad**2 * cos_fib_angle**2
    sqrt_n2_reduced = sqrt(n2_reduced)
    h_period = wavelength * (n_slab + n_clad * cos_fib_angle) / n2_reduced
    v_period = wavelength / sqrt_n2_reduced
    return h_period, v_period


def get_grating_period(
    fiber_angle: float = 13.45,
    wavelength: float = 1.55,
    neff_high: float = neff_ridge,
    neff_low: float = neff_shallow,
    n_clad: float = 1.45,
) -> float:
    """Return grating coupler period.

    based on lumerical slides.
    """
    DEG2RAD = pi / 180
    neff = (neff_high + neff_low) / 2
    return wavelength / (neff - sin(DEG2RAD * fiber_angle) * n_clad)


if __name__ == "__main__":
    pc = get_grating_period_curved()
    p = get_grating_period()
    print(pc)
    print(p)
