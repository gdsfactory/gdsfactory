"""
Compares the modes of a gdsfactory + MEEP waveguide cross-section vs a direct MPB calculation
"""

import matplotlib.pyplot as plt

from gdsfactory import add_padding
from gdsfactory.components import straight
from gdsfactory.simulation.gmeep import get_simulation
from gdsfactory.simulation.gmeep.get_port_eigenmode import get_portx_eigenmode
from gdsfactory.simulation.modes import find_modes, get_mode_solver_rib


def test_eigenmode():
    # MPB mode
    ms = get_mode_solver_rib(wg_width=0.5)
    modes = find_modes(mode_solver=ms, res=50)
    m1_MPB = modes[1]

    # MEEP monitor mode
    c = straight(length=2, width=0.5)
    c = add_padding(c.copy(), default=0, bottom=3, top=3, layers=[(100, 0)])

    sim_dict = get_simulation(
        c,
        is_3d=True,
        res=30,
        port_source_offset=-0.1,
        port_field_monitor_offset=-0.1,
        port_margin=2.5,
    )

    m1_MEEP = get_portx_eigenmode(
        sim_dict=sim_dict,
        source_index=0,
        port_name="o1",
    )

    print(m1_MPB.neff, m1_MEEP.neff)

    m1_MEEP.plot_ey()
    m1_MPB.plot_ey()
    m1_MEEP.plot_ex()
    m1_MPB.plot_ex()
    m1_MEEP.plot_ez()
    m1_MPB.plot_ez()
    plt.show()

    # # assert np.isclose(m1.neff, neff1), (m1.neff, neff1)
    # # assert np.isclose(m2.neff, neff2), (m2.neff, neff2)


if __name__ == "__main__":
    test_eigenmode()
