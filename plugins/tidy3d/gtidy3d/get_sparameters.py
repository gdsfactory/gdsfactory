import numpy as np
import tidy3d as td

from gtidy3d.run_simulation import run_simulation


def get_sparameters(sim: td.Simulation) -> np.ndarray:
    """Adapted from tidy3d examples.

    Returns full Smatrix for a component

    https://support.lumerical.com/hc/en-us/articles/360042095873-Metamaterial-S-parameter-extraction
    """

    sim = run_simulation(sim)

    def get_power(monitor):
        f, b = sim.data(monitor)["mode_amps"]
        F, B = np.abs(f) ** 2, np.abs(b) ** 2
        return F, B

    monitors = sim.monitors
    norm, _ = get_power(monitors[1])
    norm = np.squeeze(norm)

    if len(monitors) == 5:
        _, incident, reflect, top, bot = monitors
        S = np.zeros((2, 2))
        S[0, 0] = get_power(incident)[-1]
        S[1, 0] = get_power(reflect)[-1]
        S[0, 1] = get_power(top)[0]
        S[1, 1] = get_power(bot)[0]

    elif len(monitors) == 3:
        _, incident, reflect = monitors
        S = np.zeros((2, 2))
        S[0, 0] = S[1, 1] = get_power(incident)[-1]
        S[1, 0] = S[0, 1] = get_power(reflect)[-1]

    S = S / norm
    return S


if __name__ == "__main__":
    import pp
    import gtidy3d as gm

    c = pp.components.straight(length=2)
    sim = gm.get_simulation(c)
    s = get_sparameters(sim)
    print(s)
