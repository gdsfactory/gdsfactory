import numpy as np
from tidy3d import web


def get_transmission(sim, taskId, verbose=False):
    """Adapted from tidy3d examples
    https://simulation.cloud/docs/html/examples/ParameterScan.html

    Computes the transmission of a 2x2 coupler
    """
    # download results if the job has finished
    web.download_results(taskId, target_folder="out/")
    if verbose:
        web.monitor_project(taskId)
        with open("out/tidy3d.log") as f:
            print(f.read())
    sim.load_results("out/monitor_data.hdf5")

    def get_power(monitor):
        f, b = sim.data(monitor)["mode_amps"]
        F, B = np.abs(f) ** 2, np.abs(b) ** 2
        return F, B

    _, incident, reflect, top, bot = sim.monitors

    norm, _ = get_power(incident)
    norm = np.squeeze(norm)

    S = np.zeros((2, 2))
    S[0, 0] = get_power(incident)[-1]
    S[1, 0] = get_power(reflect)[-1]
    S[0, 1] = get_power(top)[0]
    S[1, 1] = get_power(bot)[0]
    S = S / norm

    split_ratio = S[0, 1] / (S[0, 1] + S[1, 1])
    efficiency = (S[0, 1] + S[1, 1]) / norm

    if verbose:
        print(f"split ratio of {(split_ratio * 100):.2f}% to top port")
        print(f"efficiency of {(efficiency * 100):.2f}% to transmission port")

    return split_ratio, efficiency


if __name__ == "__main__":
    import pp
    import matplotlib.pyplot as plt
    import gtidy3d as gm

    wg_height = 0.22
    c = pp.components.straight(length=2)
    sim, task_id = gm.get_simulation(c, is_3d=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    sim.viz_mat_2D(normal="z", position=wg_height / 2, ax=ax1)
    sim.viz_mat_2D(normal="x", ax=ax2, source_alpha=1)
    ax2.set_xlim([-3, 3])
    # plt.show()

    web.monitor_project(task_id)
    split_ratio, efficiency = get_transmission(sim, task_id, verbose=True)
