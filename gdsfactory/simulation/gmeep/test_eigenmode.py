"""
Compares the modes of a gdsfactory + MEEP waveguide cross-section vs a direct MPB calculation
"""

import matplotlib.pyplot as plt
import numpy as np

from gdsfactory import add_padding
from gdsfactory.components import straight
from gdsfactory.simulation.gmeep import get_simulation
from gdsfactory.simulation.gmeep.get_port_eigenmode import get_portx_eigenmode
from gdsfactory.simulation.modes import find_modes, get_mode_solver_rib
from gdsfactory.simulation.modes.types import Mode


def MPB_eigenmode():
    ms = get_mode_solver_rib(wg_width=0.5, sy=4, sz=4)
    modes = find_modes(mode_solver=ms, res=50)
    m1_MPB = modes[1]
    return m1_MPB


def MPB_eigenmode_toDisk():
    m1_MPB = MPB_eigenmode()
    np.save("test_data/stripWG_mpb/neff.npy", m1_MPB.neff)
    np.save("test_data/stripWG_mpb/E.npy", m1_MPB.E)
    np.save("test_data/stripWG_mpb/H.npy", m1_MPB.H)
    np.save("test_data/stripWG_mpb/y.npy", m1_MPB.y)
    np.save("test_data/stripWG_mpb/z.npy", m1_MPB.z)


def test_eigenmode(plot=False):
    """
    WARNING: Segmentation fault occurs if both ms object above and sim object exist in memory at the same time
    Instead load results from separate MPB run

    Same namespace run does not work
    # MPB mode
    # ms = get_mode_solver_rib(wg_width=0.5)
    # modes = find_modes(mode_solver=ms, res=50)
    # m1_MPB = modes[1]

    separate namespace run does not work either
    # m1_MPB = MPB_eigenmode()
    """
    # Load previously-computed waveguide results
    m1_MPB_neff = np.load("test_data/stripWG_mpb/neff.npy")
    m1_MPB_E = np.load("test_data/stripWG_mpb/E.npy")
    m1_MPB_H = np.load("test_data/stripWG_mpb/H.npy")
    m1_MPB_y = np.load("test_data/stripWG_mpb/y.npy")
    m1_MPB_z = np.load("test_data/stripWG_mpb/z.npy")
    # Package into modes object
    m1_MPB = Mode(
        mode_number=1,
        neff=m1_MPB_neff,
        wavelength=None,
        ng=None,
        E=m1_MPB_E,
        H=m1_MPB_H,
        eps=None,
        y=m1_MPB_y,
        z=m1_MPB_z,
    )

    # MEEP calculation
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

    m1_MEEP = get_portx_eigenmode(
        sim_dict=sim_dict,
        source_index=0,
        port_name="o1",
        y=m1_MPB.y,
        z=m1_MPB.z,
    )

    if plot:
        plt.figure(figsize=(10, 8), dpi=100)

        plt.subplot(3, 2, 1)
        m1_MEEP.plot_hx(show=False, operation=np.abs)
        plt.title(r"MEEP get_eigenmode \n Abs($E_x$)")

        plt.subplot(3, 2, 2)
        m1_MPB.plot_hx(show=False, operation=np.abs)
        plt.title(r"MPB find_modes \n Abs($E_x$)")

        plt.subplot(3, 2, 3)
        m1_MEEP.plot_hy(show=False, operation=np.abs)

        plt.subplot(3, 2, 4)
        m1_MPB.plot_hy(show=False, operation=np.abs)

        plt.subplot(3, 2, 5)
        m1_MEEP.plot_hz(show=False, operation=np.abs)

        plt.subplot(3, 2, 6)
        m1_MPB.plot_hz(show=False, operation=np.abs)

        plt.tight_layout()
        plt.show()

    # # assert np.isclose(m1.neff, neff1), (m1.neff, neff1)
    # # assert np.isclose(m2.neff, neff2), (m2.neff, neff2)


if __name__ == "__main__":
    # MPB_eigenmode_toDisk()
    test_eigenmode(plot=True)
