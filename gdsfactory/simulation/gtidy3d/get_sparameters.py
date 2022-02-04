import numpy as np
import tidy3d as td

from gdsfactory.simulation.gtidy3d.get_results import get_results


def get_sparameter(sim: td.Simulation, p1: str = "o1", p2: str = "o2") -> np.ndarray:
    """Return Component sparameters
    Adapted from tidy3d examples.
    """

    r = get_results(sim).result()

    def get_amplitude(monitor):
        f, b = r.monitor_data["amps"]
        return np.squeeze(f), np.squeeze(b)

    # monitor_data = r.monitor_data
    # n = len(monitor_data) - 1

    S = {}

    # S = np.zeros((n, n), dtype=np.complex128)
    # for i, monitor_i in enumerate(monitor_data):
    #     for j, monitor_j in enumerate(monitor_data):
    #         if i > 0 and j > 0:
    #             if monitor_i.name.startswith("W"):
    #                 ai, bi = get_amplitude(monitor_i)
    #             else:
    #                 bi, ai = get_amplitude(monitor_i)
    #             if monitor_j.name.startswith("W"):
    #                 aj, bj = get_amplitude(monitor_j)
    #             else:
    #                 bj, aj = get_amplitude(monitor_j)
    #             S[i - i, j - 1] = bi / aj
    # if len(monitor_data) == 5:
    #     a = monitor_data[p1]
    #     b = monitor_data[p2]

    #     S[0, 0] = get_amplitude(incident)[-1]
    #     S[1, 0] = get_amplitude(reflect)[-1]

    # elif len(monitor_data) == 3:
    #     _, incident, reflect = monitor_data
    #     S[0, 0] = S[1, 1] = get_amplitude(incident)[-1]
    #     S[1, 0] = S[0, 1] = get_amplitude(reflect)[-1]
    return S


if __name__ == "__main__":
    import gdsfactory as gf
    import gdsfactory.simulation.gtidy3d as gt

    c = gf.components.straight(length=2)
    sim = gt.get_simulation(c)
    # s = get_sparameters(sim)
    # print(s)

    r = get_results(sim).result()

    def get_amplitude(monitor):
        f, b = r.data(monitor)["amps"]
        # return f, b
        return np.squeeze(f), np.squeeze(b)

    monitor_data = r.monitor_data

    n = len(monitor_data) - 1
    S = np.zeros((n, n), dtype=np.complex128)

    # for i, monitor_i in enumerate(monitor_data):
    #     for j, monitor_j in enumerate(monitor_data):
    #         if i > 0 and j > 0:
    #             # ai, bi = get_amplitude(monitor_i)
    #             # aj, bj = get_amplitude(monitor_j)
    #             # S[i-1, j-1] = bi / aj
    #             if monitor_i.name.startswith("W"):
    #                 ai, bi = get_amplitude(monitor_i)
    #             else:
    #                 bi, ai = get_amplitude(monitor_i)
    #             if monitor_j.name.startswith("W"):
    #                 aj, bj = get_amplitude(monitor_j)
    #             else:
    #                 bj, aj = get_amplitude(monitor_j)
    #             S[i - 1, j - 1] = bi / aj

    _, incident, reflect = monitor_data
    a1, b1 = get_amplitude(incident)
    b2, a2 = get_amplitude(reflect)
    S[0, 0] = S[1, 1] = b1 / a1  # S11
    S[1, 0] = S[0, 1] = b2 / a1  # S12
