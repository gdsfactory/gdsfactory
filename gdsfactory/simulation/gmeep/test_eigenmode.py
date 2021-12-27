"""
Compares the modes of a gdsfactory + MEEP waveguide cross-section vs a direct MPB calculation
"""

# import matplotlib.pyplot as plt

from gdsfactory import add_padding
from gdsfactory.components import straight
from gdsfactory.simulation.gmeep import get_simulation
from gdsfactory.simulation.gmeep.plotting import plot_eigenmode


def test_eigenmode():
    # MPB mode
    # ms = get_mode_solver_rib(wg_width=0.5)
    # modes = find_modes(mode_solver=ms, res=50)
    # m1 = modes[1]
    # m2 = modes[2]

    # MEEP monitor mode
    c = straight(length=2, width=0.5)
    c = add_padding(c.copy(), default=0, bottom=3, top=3, layers=[(100, 0)])

    sim_dict = get_simulation(
        c,
        is_3d=True,
        res=50,
        port_source_offset=-0.1,
        port_field_monitor_offset=-0.1,
        port_margin=2.5,
    )
    sim = sim_dict["sim"]

    # mode = get_port_eigenmode(sim, sim_dict["sources"][0], sim_dict["monitors"]["o1"])
    plot_eigenmode(sim, sim_dict["sources"][0], sim_dict["monitors"]["o1"])

    # assert np.isclose(m1.neff, neff1), (m1.neff, neff1)
    # assert np.isclose(m2.neff, neff2), (m2.neff, neff2)


if __name__ == "__main__":
    test_eigenmode()
