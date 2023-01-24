import numpy as np


def free_spectral_range(n: float, length: float, wavelength: float) -> float:
    """Returns Free Spectral range in um.

    Args:
        n: index.
        length: in um.
        wavelength: in um.

    """
    return wavelength**2 / (n * length)


def directional_coupler_lc(
    wavelength_nm: int, n_eff_1: float, n_eff_2: float, power_ratio: float = 1.0
) -> float:
    """Calculates directional coupler coherence length (100% power transfer).

    Args:
        wavelength_nm: The wavelength in [nm] the directional coupler should operate at.
        n_eff_1: n_eff of the fundamental (even) supermode of the directional coupler.
        n_eff_2: n_eff of the first-order (odd) supermode of the directional coupler.
        power_ratio: p2/p1.

    """
    wavelength_m = wavelength_nm * 1.0e-9
    dn_eff = (n_eff_1 - n_eff_2).real
    lc_m = wavelength_m / (np.pi * dn_eff) * np.arcsin(np.sqrt(power_ratio))
    return lc_m * 1.0e6


def grating_coupler_period(
    wavelength: float,
    n_eff: float,
    n_clad: float,
    incidence_angle_deg: float,
    diffration_order: int = 1,
) -> float:
    """Returns period needed for a grating coupler.

    Args:
        wavelength: target wavelength for the grating coupler.
        n_eff: effective index of the mode of a waveguide with the width of the grating coupler.
        n_clad refractive index of the cladding.
        incidence_angle_deg: incidence angle the grating coupler should operate in degrees.
        diffration_order: grating order the coupler should work at. Default is 1st order (1).

    Returns:
        float: The period needed for the grating coupler in the same units as wavelength
    """
    k0 = 2.0 * np.pi / wavelength
    beta = n_eff.real * k0
    n_inc = n_clad

    return (2.0 * np.pi * diffration_order) / (
        beta - k0 * n_inc * np.sin(np.radians(incidence_angle_deg))
    )


if __name__ == "__main__":
    fsr = free_spectral_range(n=2.4, length=20, wavelength=1.55)
    print(fsr * 1e3)
