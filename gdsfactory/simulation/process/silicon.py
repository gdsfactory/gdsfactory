"""Temporary file to hold silicon material properties until better material database implemented.

TODO:
    Material class
    Pint for units
"""
import numpy as np

kB = 1.380649e-23  # m2 kg s-2 K-1, boltzmann constant
kB_eV = 8.617333262e-5  # eV K-1, boltzmann constant
h = 6.62607015e-34  # m2 kg s-1, Planck constant
m0 = 9.1093837e-31  # kg, electron mass


def Eg(T: float):
    """Temperature-dependent bandgap.

    Arguments:
        T: temperature (K)

    Returns:
        Bandgap (eV)
    """
    return 1.1785 - 9.025 * 1e-5 * T - 3.05 * 1e-7 * T**2


def mn(T: float):
    """Electron (relative) effective mass.

    Arguments:
        T: temperature (K)

    Returns:
        Effective mass (units of m0)
    """
    return 1.045 + 4.5 * 1e-4 * T


def mp(T: float):
    """Hole (relative) effective mass.

    Ref: https://www.ioffe.ru/SVA/NSM/Semicond/Si/bandstr.html#:~:text=mcd%20%3D%201.18mo,of%20the%20density%20of%20states.
    TODO: find better T-dep

    Maybe? D. M. Riffe, "Temperature dependence of silicon carrier effective masses with application to femtosecond reflectivity measurements," J. Opt. Soc. Am. B 19, 1092-1100 (2002)

    Arguments:
        T: temperature (K) (fixed to 300K to stay positive)

    Returns:
        Effective mass (units of m0)
    """
    return 0.523 + 1.4 * 1e-3 * 300 - 1.48 * 1e-6 * 300**2


def N(T: float, m: float):
    """Conduction band density of states.

    Arguments:
        T: temperature (K)
        m: carrier rel. effective mass (unitless)

    Returns:
        DOS (unitless)
    """
    return 2 * np.power((2 * np.pi * kB * T * m * m0) / (h**2), 3 / 2)


def ni(T: float):
    """Intrinsic carrier concentration.

    Arguments:
        T: temperature (K)

    Returns:
        ni: intrinsic carrier concentration (cm-3)
    """
    return (
        np.sqrt(N(T=T, m=mn(T)) * N(T=T, m=mp(T)))
        * np.exp(-1 * Eg(T) / (2 * kB_eV * T))
        * 1e-6
    )


if __name__ == "__main__":
    # print(2*kB_eV*300)
    print(Eg(T=1000), Eg(T=300))
    print(np.log10(ni(T=800)))

    # Ts = np.linspace(800,1100,100) + 150

    # for T in Ts:
    #     print(mp(T))
    #     # print(N(T, m=mp(T)))
