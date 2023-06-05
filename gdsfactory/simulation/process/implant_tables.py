"""
Reference: Selberherr, S. (1984). Process Modeling. In: Analysis and Simulation of Semiconductor Devices. Springer, Vienna. https://doi.org/10.1007/978-3-7091-8752-4_3
"""
from pathlib import Path

import numpy as np
from scipy import interpolate, optimize

"""um length units, energy in keV"""
depth_in_silicon = {
    "boron": np.poly1d([-3.308e-6, 3.338e-3, 0], variable="E"),
    "phosphorus": np.poly1d([1.290e-9, -2.743e-7, 1.259e-3, 0], variable="E"),
    "antimony": np.poly1d(
        [4.028e-13, -3.056e-10, 8.372e-8, -1.013e-5, 8.887e-4, 0], variable="E"
    ),
    "arsenic": np.poly1d(
        [4.608e-13, -3.442e-10, 9.067e-8, -1.022e-5, 9.818e-4, 0], variable="E"
    ),
}

straggle_in_silicon = {
    "boron": np.poly1d(
        [5.525e-13, -4.545e-10, 1.403e-7, -2.086e-5, 1.781e-3, 0], variable="E"
    ),
    "phosphorus": np.poly1d(
        [-2.252e-11, 1.371e-8, -3.161e-6, 6.542e-4, 0], variable="E"
    ),
    "antimony": np.poly1d(
        [1.084e-13, -8.310e-11, 2.311e-8, -2.885e-6, 2.674e-4, 0], variable="E"
    ),
    "arsenic": np.poly1d(
        [1.601e-13, -1.202e-10, 3.235e-8, -3.820e-6, 3.652e-4, 0], variable="E"
    ),
}


module_path = Path(__file__).parent
skew_data = {
    "boron": np.genfromtxt(
        module_path / "skew/boron_si_skew.csv", delimiter=" ", dtype=float
    ),
    "phosphorus": np.genfromtxt(
        module_path / "skew/phosphorus_si_skew.csv", delimiter=" ", dtype=float
    ),
    "antimony": np.genfromtxt(
        module_path / "skew/antimony_si_skew.csv", delimiter=" ", dtype=float
    ),
    "arsenic": np.genfromtxt(
        module_path / "skew/arsenic_si_skew.csv", delimiter=" ", dtype=float
    ),
}

skew_in_silicon = {
    "boron": interpolate.interp1d(
        x=skew_data["boron"][:, 0], y=skew_data["boron"][:, 1], fill_value="extrapolate"
    ),
    "phosphorus": interpolate.interp1d(
        x=skew_data["phosphorus"][:, 0],
        y=skew_data["phosphorus"][:, 1],
        fill_value="extrapolate",
    ),
    "antimony": interpolate.interp1d(
        x=skew_data["antimony"][:, 0],
        y=skew_data["antimony"][:, 1],
        fill_value="extrapolate",
    ),
    "arsenic": interpolate.interp1d(
        x=skew_data["arsenic"][:, 0],
        y=skew_data["arsenic"][:, 1],
        fill_value="extrapolate",
    ),
}


def silicon_gaussian_profile(
    dopant: str, dose: float, E: float, z: np.array = np.linspace(0, 1, 1000)
):
    """
    Returns gaussian implantation profile for dopant in silicon.

    Arguments:
        dopant: str name of implant
        dose: implant dose per unit area (/cm2)
        E: energy of implant (keV)
        z: depth coordinate (um)

    Returns:
        C(z): ion distribution as a function of substrate depth (ions/cm3)
        (Need to convert straggle from um to cm)
    """
    z0 = depth_in_silicon[dopant](E)
    dz = straggle_in_silicon[dopant](E)
    return (
        dose
        * np.reciprocal(np.sqrt(2 * np.pi) * dz * 1e-4)
        * np.exp(-((z - z0) ** 2) / (2 * dz**2))
    )


def silicon_skewed_gaussian_profile(
    dopant: str, dose: float, E: float, z: np.array = np.linspace(0, 1, 1000)
):
    """
    Returns skewed two half-gaussian implantation profile for dopant in silicon. Valid for |skew| <~ 1.

    Arguments:
        dopant: str name of implant
        dose: implant dose per unit area (/cm2)
        E: energy of implant (keV)
        z: depth coordinate (um)

    Returns:
        C(z): ion distribution as a function of substrate depth (ions/cm3)
        (Need to convert straggle from um to cm)
    """
    Rp = depth_in_silicon[dopant](E)
    sigmap = straggle_in_silicon[dopant](E)
    gamma1 = skew_in_silicon[dopant](E)

    def Rp_eq(Rm, sigma1, sigma2):
        return Rm + np.sqrt(2 / np.pi) * (sigma2 - sigma1)

    def sigmap_eq(Rm, sigma1, sigma2):
        return np.sqrt(
            (sigma1**2 - sigma1 * sigma2 + sigma2**2)
            - 2 / np.pi * (sigma2 - sigma1) ** 2
        )

    def gamma1_eq(Rm, sigma1, sigma2):
        return (
            np.sqrt(2 / np.pi)
            * (sigma2 - sigma1)
            * (
                (4 / np.pi - 1) * (sigma1**2 + sigma2**2)
                + (3 - 8 / np.pi) * sigma1 * sigma2
            )
            / sigmap_eq(Rm, sigma1, sigma2) ** 3
        )

    def system(x):
        Rm, sigma1, sigma2 = x
        return [
            Rp_eq(Rm, sigma1, sigma2) - Rp,
            sigmap_eq(Rm, sigma1, sigma2) - sigmap,
            gamma1_eq(Rm, sigma1, sigma2) - gamma1,
        ]

    Rm, sigma1, sigma2 = optimize.fsolve(system, [Rp, 0.9 * sigmap, 1.1 * sigmap])

    return (
        dose
        * 2
        * np.reciprocal(np.sqrt(2 * np.pi) * (sigma1 + sigma2) * 1e-4)
        * np.where(
            z < Rm,
            np.exp(-((z - Rm) ** 2) / (2 * sigma1**2)),
            np.exp(-((z - Rm) ** 2) / (2 * sigma2**2)),
        )
    )


if __name__ == "__main__":
    energies = [20, 40, 60, 80, 100, 120, 140, 160]
    z = np.linspace(0, 0.25, 1000)

    import matplotlib.pyplot as plt

    lower_lim = 0
    for E in energies:
        c = silicon_skewed_gaussian_profile("arsenic", dose=1e15, E=E, z=z)
        plt.semilogy(z, c, label=E)
        if c[0] > lower_lim:
            lower_lim = c[0]

    plt.ylim([lower_lim, 1e17])
    plt.xlim([0, 0.2])
    plt.legend()

    plt.show()
