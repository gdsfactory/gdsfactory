import numpy as np
import tidy3d as td

from gdsfactory.simulation.tidy3d.run_simulation import run_simulation


def get_sparameters(sim: td.Simulation) -> np.ndarray:
    """Adapted from tidy3d examples.

    Returns full Smatrix for a component

    https://support.lumerical.com/hc/en-us/articles/360042095873-Metamaterial-S-parameter-extraction
    """

    sim = run_simulation(sim).result()

    def get_amplitude(monitor):
        f, b = sim.data(monitor)["mode_amps"]
        return np.squeeze(f), np.squeeze(b)

    monitors = sim.monitors

    n = len(monitors) - 1
    S = np.zeros((n, n), dtype=np.complex128)

    # for i, monitor_i in enumerate(monitors):
    #     for j, monitor_j in enumerate(monitors):
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

    if len(monitors) == 5:
        _, incident, reflect, top, bot = monitors
        S[0, 0] = get_amplitude(incident)[-1]
        S[1, 0] = get_amplitude(reflect)[-1]
        S[0, 1] = get_amplitude(top)[0]
        S[1, 1] = get_amplitude(bot)[0]

    elif len(monitors) == 3:
        _, incident, reflect = monitors
        S[0, 0] = S[1, 1] = get_amplitude(incident)[-1]
        S[1, 0] = S[0, 1] = get_amplitude(reflect)[-1]
    return S


if __name__ == "__main__":
    import gdsfactory as gf
    import gdsfactory.simulation.tidy3d as gm

    c = gf.components.straight(length=2)
    sim = gm.get_simulation(c)
    # s = get_sparameters(sim)
    # print(s)

    sim = run_simulation(sim).result()

    def get_amplitude(monitor):
        f, b = sim.data(monitor)["mode_amps"]
        # return f, b
        return np.squeeze(f), np.squeeze(b)

    monitors = sim.monitors

    n = len(monitors) - 1
    S = np.zeros((n, n), dtype=np.complex128)

    # for i, monitor_i in enumerate(monitors):
    #     for j, monitor_j in enumerate(monitors):
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

    _, incident, reflect = monitors
    a1, b1 = get_amplitude(incident)
    b2, a2 = get_amplitude(reflect)
    S[0, 0] = S[1, 1] = b1 / a1  # S11
    S[1, 0] = S[0, 1] = b2 / a1  # S12
