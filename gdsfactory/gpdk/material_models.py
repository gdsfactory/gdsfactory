"""Locally evaluated optical equations with coefficients from primary papers."""

from math import exp

import numpy as np
import numpy.typing as npt

HC_EV_UM = 1.2398419843320026

LI_WAVELENGTHS_UM = np.array(
    [
        1.20,
        1.22,
        1.24,
        1.26,
        1.28,
        1.30,
        1.32,
        1.34,
        1.36,
        1.38,
        1.40,
        1.45,
        1.50,
        1.55,
        1.60,
        1.65,
        1.70,
        1.80,
        1.90,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
        4.00,
        5.00,
        6.00,
        7.00,
        8.00,
        9.00,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
    ]
)


def silicon_li_index(
    wavelength_um: npt.ArrayLike,
    temperature_kelvin: float = 293.0,
) -> npt.NDArray[np.float64]:
    """Li (1980), Eq. (22), using the same coefficients and sampling as gsim."""
    temperature = temperature_kelvin
    if temperature <= 293:
        dilation = (
            -2.1e-4
            - 4.149e-7 * temperature
            - 4.620e-10 * temperature**2
            + 1.482e-11 * temperature**3
        )
    else:
        dilation = (
            -7.1e-4
            + 1.887e-6 * temperature
            + 1.934e-9 * temperature**2
            - 4.544e-13 * temperature**3
        )
    epsilon = (
        11.4445
        + 2.7739e-4 * temperature
        + 1.7050e-6 * temperature**2
        - 8.1347e-10 * temperature**3
    )
    strength = 0.8948 + 4.3977e-4 * temperature + 7.3835e-8 * temperature**2
    wavelength = np.asarray(wavelength_um, dtype=float)
    return np.asarray(np.sqrt(epsilon + exp(-3 * dilation) * strength / wavelength**2))


def _drude_lorentz_index(
    wavelength_um: npt.ArrayLike,
    *,
    eps_inf: float,
    plasma_energy_ev: float,
    drude_width_ev: float,
    oscillators: tuple[tuple[float, float, float], ...],
) -> npt.NDArray[np.complex128]:
    """Evaluate passive n+i*k; oscillators are (delta-eps, energy, width).

    Epsilon = eps_inf - Ep²/[E(E+i*Gamma)]
              + sum(delta-eps * Ej²/[Ej²-E²-i*gammaj*E]).
    All energies and linewidths use eV. The convention is exp(-i*omega*t).
    """
    energy = HC_EV_UM / np.asarray(wavelength_um, dtype=float)
    permittivity = eps_inf - plasma_energy_ev**2 / (
        energy * (energy + 1j * drude_width_ev)
    )
    for strength, resonance, width in oscillators:
        permittivity += (
            strength * resonance**2 / (resonance**2 - energy**2 - 1j * width * energy)
        )
    return np.asarray(np.sqrt(permittivity), dtype=complex)


def aluminum_index(wavelength_um: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Rakic et al. (1998), Eq. (1), Tables 1-2, DOI 10.1364/AO.37.005271."""
    plasma = 14.98
    # Convert the paper's fj*Ep² numerator to delta-eps*Ej².
    return _drude_lorentz_index(
        wavelength_um,
        eps_inf=1,
        plasma_energy_ev=np.sqrt(0.523) * plasma,
        drude_width_ev=0.047,
        oscillators=tuple(
            (strength * plasma**2 / resonance**2, resonance, width)
            for strength, resonance, width in (
                (0.227, 0.162, 0.333),
                (0.050, 1.544, 0.312),
                (0.166, 1.808, 1.351),
                (0.030, 3.473, 3.382),
            )
        ),
    )


def tin_index(wavelength_um: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Bower et al. (2020), SI Table S5.1 TiN/Si, DOI 10.1021/acsami.0c10570."""
    return _drude_lorentz_index(
        wavelength_um,
        eps_inf=3.558,
        plasma_energy_ev=7.44,
        drude_width_ev=0.416,
        oscillators=((3.819, 4.802, 2.4), (0.997, 2.024, 1.617)),
    )
