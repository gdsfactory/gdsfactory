import meep as mp


def plot_xsection(sim, center=(0, 0, 0), size=(0, 2, 2)):
    """
    sim: simulation object
    """
    sim.plot2D(output_plane=mp.Volume(center=center, size=size))
    # plt.colorbar()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory import add_padding
    from gdsfactory.components import straight
    from gdsfactory.simulation.gmeep import get_simulation

    c = straight(length=2, width=0.45)
    c = add_padding(c.copy(), default=0, bottom=3, top=3, layers=[(100, 0)])

    sim_dict = get_simulation(
        c,
        is_3d=True,
        port_source_offset=-0.1,
        port_margin=2.5,
    )

    plt.subplot(1, 2, 1)
    plot_xsection(
        sim_dict["sim"],
        center=sim_dict["monitors"]["o1"].regions[0].center,
        size=sim_dict["monitors"]["o1"].regions[0].size,
    )

    from gdsfactory.simulation.modes import find_modes, get_mode_solver_rib

    ms = get_mode_solver_rib(wg_width=0.45, sy=5, sz=3)
    modes = find_modes(mode_solver=ms, res=50)

    plt.subplot(1, 2, 2)
    modes[1].plot_eps()
    plt.show()
