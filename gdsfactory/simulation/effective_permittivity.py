"""Calculate the effective refractive index for a 1D mode."""

from typing import List

import numpy as np
from scipy.optimize import fsolve
from typing_extensions import Literal


def calculate_effective_permittivity(
    epsilon_film: float,
    epsilon_substrate: float,
    epsilon_cladding: float,
    thickness: float,
    wavelength: float,
    polarization: Literal["te", "tm"],
) -> List[float]:
    """
    Calculate the effective refractive index for a 1D mode.

    .. code::

        -----------------      |
        epsilon_cladding      inf
        -----------------      |
        epsilon_film        thickness
        -----------------      |
        epsilon_substrate     inf
        -----------------      |

    Args:
        epsilon_film: Relative permittivity of the film.
        epsilon_substrate: Relative permittivity of the substrate.
        epsilon_cladding: Relative permittivity of the cladding.
        thickness: Thickness of the film.
        wavelength: Wavelength.
        polarization: Either "te" or "tm".

    Returns:
        List of effective permittivity.
    """
    if polarization == "te":
        tm = False
    elif polarization == "tm":
        tm = True
    else:
        raise ValueError('Polarization must be "te" or "tm"')

    k_0 = 2 * np.pi / wavelength

    def k_f(e_eff):
        return k_0 * np.sqrt(epsilon_film - e_eff) / (epsilon_film if tm else 1)

    def k_s(e_eff):
        return (
            k_0 * np.sqrt(e_eff - epsilon_substrate) / (epsilon_substrate if tm else 1)
        )

    def k_c(e_eff):
        return k_0 * np.sqrt(e_eff - epsilon_cladding) / (epsilon_cladding if tm else 1)

    def objective(e_eff):
        return 1 / np.tan(k_f(e_eff) * thickness) - (
            k_f(e_eff) ** 2 - k_s(e_eff) * k_c(e_eff)
        ) / (k_f(e_eff) * (k_s(e_eff) + k_c(e_eff)))

    # scan roughly for indices
    # use a by 1e-10 smaller search area to avoid division by zero
    x = np.linspace(
        min(epsilon_substrate, epsilon_cladding) + 1e-10, epsilon_film - 1e-10, 1000
    )
    indices_temp = x[np.abs(objective(x)) < 0.1]
    if not len(indices_temp):
        return []

    # and then use fsolve to get exact indices
    indices_temp = fsolve(objective, indices_temp)

    # then make the indices unique
    indices = []
    for index in indices_temp:
        if not any(np.isclose(index, i, atol=1e-7) for i in indices):
            indices.append(index)

    return indices


if __name__ == "__main__":
    print(
        calculate_effective_permittivity(
            3.4777**2, 1.444**2, 1.444**2, 0.22e-6, 1.55e-6, "te"
        )
    )
