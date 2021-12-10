import numpy as np
import tidy3d as td

from gdsfactory.simulation.tidy3d.get_sparameters import get_sparameters


def get_coupling(sim: td.Simulation) -> np.ndarray:
    """Adapted from tidy3d examples
    https://simulation.cloud/docs/html/examples/ParameterScan.html

    Computes the transmission  and efficiency of a 2x2 coupler
    """
    S = get_sparameters(sim)
    split_ratio_top = S[0, 1] / (S[0, 1] + S[1, 1])
    split_ratio_bot = S[1, 1] / (S[0, 1] + S[1, 1])
    print(f"split ratio of {(split_ratio_top * 100):.2f}% to top port")
    print(f"split ratio of {(split_ratio_bot * 100):.2f}% to bot port")

    # efficiency = S[0, 1] + S[1, 1]
    # print(f"efficiency of {(efficiency * 100):.2f}% to transmission port")
    return S


if __name__ == "__main__":
    import gdsfactory as gf
    import gdsfactory.simulation.tidy3d as gm

    c = gf.components.coupler(gap=0.1, length=6.0)
    sim = gm.get_simulation(c)

    s = get_coupling(sim)
