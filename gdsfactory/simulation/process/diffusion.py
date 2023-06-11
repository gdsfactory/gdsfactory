"""
Reference: Selberherr, S. (1984). Process Modeling. In: Analysis and Simulation of Semiconductor Devices. Springer, Vienna. https://doi.org/10.1007/978-3-7091-8752-4_3
"""
from typing import Optional

import numpy as np

from gdsfactory.simulation.process.silicon import ni

diffusion_in_silicon = {
    "boron": {
        "D00": 0.037,  # cm2s-1
        "Ea0": 3.46,  # eV
        "D0-": 0.0,
        "Ea-": 0.0,
        "D0=": 0.0,
        "Ea=": 0.0,
        "D0+": 0.72,
        "Ea+": 3.46,
    },
    "phosphorus": {
        "D00": 3.85,  # cm2s-1
        "Ea0": 3.66,  # eV
        "D0-": 4.44,
        "Ea-": 4.0,
        "D0=": 44.20,
        "Ea=": 4.37,
        "D0+": 0.0,
        "Ea+": 0.0,
    },
    "antimony": {
        "D00": 0.214,  # cm2s-1
        "Ea0": 3.65,  # eV
        "D0-": 15.0,
        "Ea-": 4.08,
        "D0=": 0.0,
        "Ea=": 0.0,
        "D0+": 0.0,
        "Ea+": 0.0,
    },
    "arsenic": {
        "D00": 0.066,  # cm2s-1
        "Ea0": 3.44,  # eV
        "D0-": 12.0,
        "Ea-": 4.05,
        "D0=": 0.0,
        "Ea=": 0.0,
        "D0+": 0.0,
        "Ea+": 0.0,
    },
}


def D(
    dopant: str,
    T: float,
    n: Optional[float] = None,
    p: Optional[float] = None,
):
    """
    Diffusion coefficient of dopants in silicon.

    Arguments:
        dopant: dopant atom name
        T: temperature (C)
        n: donor concentration (/cm3), defaults to ni
        p: acceptor concentration (/cm3), defaults to ni
    """
    kB = 8.617333262e-5  # eV K-1
    T_kelvin = T + 273.15
    if n is None:
        n = ni(T_kelvin)
    if p is None:
        p = ni(T_kelvin)
    Di0 = diffusion_in_silicon[dopant]["D00"] * np.exp(
        -1 * diffusion_in_silicon[dopant]["Ea0"] / (kB * T_kelvin)
    )
    Din = diffusion_in_silicon[dopant]["D0-"] * np.exp(
        -1 * diffusion_in_silicon[dopant]["Ea-"] / (kB * T_kelvin)
    )
    Dinn = diffusion_in_silicon[dopant]["D0="] * np.exp(
        -1 * diffusion_in_silicon[dopant]["Ea="] / (kB * T_kelvin)
    )
    Dip = diffusion_in_silicon[dopant]["D0+"] * np.exp(
        -1 * diffusion_in_silicon[dopant]["Ea+"] / (kB * T_kelvin)
    )
    return (
        Di0
        + Din * n / ni(T_kelvin)
        + Dinn * (n / ni(T_kelvin)) ** 2
        + Dip * p / ni(T_kelvin)
    )


def silicon_diffused_gaussian_profile(
    dopant: str,
    dose: float,
    E: float,
    t: float,
    T: float,
    z: np.array = np.linspace(0, 1, 1000),
    # x: np.array = np.linspace(-5,5,1000),
):
    """
    Returns diffused gaussian implantation profile for dopant in silicon.

    Arguments:
        dopant: str name of implant
        dose: implant dose per unit area (/cm2)
        E: energy of implant (keV)
        t: diffusion time (s)
        T: temperature applied during diffusion time (C)
        z: depth coordinate (um)

    Returns:
        C(z): ion distribution as a function of substrate position and depth (ions/cm3)
    """
    from gdsfactory.simulation.process.implant_tables import (
        depth_in_silicon,
        straggle_in_silicon,
    )

    z0 = depth_in_silicon[dopant](E)
    dz = straggle_in_silicon[dopant](E)
    Dconst = D(dopant=dopant, T=T) * 1e8  # convert from cm2/s to to um2/s

    # 2D functions (TODO)
    # a = np.mean(x)
    # xx, zz = np.meshgrid(x, z)
    # def H(z,t):
    # return np.exp(-(z - z0)**2 / (2 * dz**2 + 4*Dconst*t)) * special.erf(-z0/dz * np.sqrt(Dconst*t/(dz**2 + 2*Dconst*t)) - np.sqrt(dz**2 / (4*Dconst*t*(dz**2 + 2*Dconst*t))) * z)
    # return dose / (4 * np.sqrt(2 * np.pi * (dz**2 + 2*Dconst*t))) * (H(z,t) + H(-1*z,t)) * special.erf((a - xx)/(2 * np.sqrt(Dconst * t)))

    return (
        dose
        / (np.sqrt(2 * np.pi * (dz**2 + 2 * Dconst * t)))
        * np.exp(-((z - z0) ** 2) / (dz**2 + 2 * Dconst * t))
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Ts = [800,850,900,950,1000,1050]
    # conc = np.logspace(18,21,100)
    # for T in Ts:
    #     plt.loglog(conc, D("phosphorus", T, n=conc, p=conc), label=T)
    # plt.xlabel("Acceptor concentration (cm-3)")
    # plt.ylabel("Diffusivity (cm2 s-1)")
    # plt.title("Intrinsic diffusivity (n=p=ni)")
    # plt.legend()
    # plt.show()

    for t in [0, 60, 5 * 60, 10 * 60]:
        conc = silicon_diffused_gaussian_profile(
            dopant="phosphorus",
            dose=1e12,
            E=100,
            t=t,
            T=1000,
            z=np.linspace(0, 0.6, 1000),
        )
        plt.plot(conc, label="t")
    plt.title("Phosphorus 1E12 ions/cm2, 100keV, 1000C anneal")
    plt.xlabel("depth (um)")
    plt.ylabel("ion concentration (cm-3)")
    plt.legend(title="Diffusion time (s)")
    plt.show()
