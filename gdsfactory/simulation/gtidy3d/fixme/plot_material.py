import matplotlib as mpl
import matplotlib.pyplot as plt

import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt

MATERIAL_NAME_TO_TIDY3D = {
    # "si": 3.47,
    # "sio2": 1.44,
    # "sin": 2.0,
    "si": "cSi",
    "sio2": "SiO2",
    "sin": "Si3N4",
}
if __name__ == "__main__":
    c = gf.components.straight()
    sim = gt.get_simulation(
        c, plot_modes=False, material_name_to_tidy3d=MATERIAL_NAME_TO_TIDY3D
    )

    # plot_simulation(sim)

    fig = plt.figure(figsize=(11, 4))
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    sim.plot_eps(z=0.0, ax=ax1)
    sim.plot_eps(x=0.0, ax=ax2)
    plt.show()
