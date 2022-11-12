"""Compares the modes of a gdsfactory + MEEP waveguide cross-section vs a
direct MPB calculation."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import gdsfactory as gf
from gdsfactory.components import straight
from gdsfactory.simulation.gmeep import get_simulation
from gdsfactory.simulation.gmeep.get_port_eigenmode import get_port_2Dx_eigenmode
from gdsfactory.simulation.modes import find_modes_waveguide, get_mode_solver_rib
from gdsfactory.simulation.modes.types import Mode


def lumerical_parser(E_1D, H_1D, y_1D, z_1D, res=50, z_offset=0.11 * 1e-6):
    """Converts 1D arrays of fields to 2D arrays according to positions.

    Lumerical data is in 1D arrays, and over a nonregular mesh

    Args
        E_1D: E array from Lumerical.
        H_1D: H array from Lumerical.
        y_1D: y array from Lumerical.
        z_1D: z array from Lumerical.
        res: desired resolution.
        z_offset: z offset to move the fields.

    """
    # Make regular grid from resolution and range of domain
    y_1D = y_1D[...].flatten()
    z_1D = z_1D[...].flatten()
    ny = int(np.max(y_1D) - np.min(y_1D) * 1e6 * res)
    nz = int(np.max(z_1D) - np.min(z_1D) * 1e6 * res)
    y = np.linspace(np.min(y_1D), np.max(y_1D), ny) * 1e6
    z = np.linspace(np.min(z_1D), np.max(z_1D), nz) * 1e6
    yy, zz = np.meshgrid(y, z)

    # Generates points parameter ((y,z) array) for griddata
    points = np.zeros([len(E_1D[...][0, :]), 2])
    i = 0
    for j in range(len(z_1D)):
        for k in range(len(y_1D)):
            points[i, 0] = y_1D[k] * 1e6
            points[i, 1] = z_1D[j] * 1e6
            i += 1

    # Get interpolated field values
    E = np.zeros([ny, nz, 1, 3], dtype=np.cdouble)
    H = np.zeros([ny, nz, 1, 3], dtype=np.cdouble)
    E[:, :, 0, 0] = griddata(
        points,
        E_1D[...][0, :]["real"] + 1j * E_1D[...][0, :]["imag"],
        (zz, yy),
        method="cubic",
    )
    E[:, :, 0, 1] = griddata(
        points,
        E_1D[...][1, :]["real"] + 1j * E_1D[...][1, :]["imag"],
        (zz, yy),
        method="cubic",
    )
    E[:, :, 0, 2] = griddata(
        points,
        E_1D[...][2, :]["real"] + 1j * E_1D[...][2, :]["imag"],
        (zz, yy),
        method="cubic",
    )
    H[:, :, 0, 0] = griddata(
        points,
        H_1D[...][0, :]["real"] + 1j * H_1D[...][0, :]["imag"],
        (zz, yy),
        method="cubic",
    )
    H[:, :, 0, 1] = griddata(
        points,
        H_1D[...][1, :]["real"] + 1j * H_1D[...][1, :]["imag"],
        (zz, yy),
        method="cubic",
    )
    H[:, :, 0, 2] = griddata(
        points,
        H_1D[...][2, :]["real"] + 1j * H_1D[...][2, :]["imag"],
        (zz, yy),
        method="cubic",
    )

    return E, H, y, z


def MPB_eigenmode():
    ms = get_mode_solver_rib(wg_width=0.45, sy=6, sz=6)
    modes = find_modes_waveguide(mode_solver=ms, res=50)
    m1_MPB = modes[1]
    m2_MPB = modes[2]
    return m1_MPB, m2_MPB


def MPB_eigenmode_toDisk() -> None:
    m1_MPB, m2_MPB = MPB_eigenmode()
    np.save("test_data/stripWG_mpb/neff1.npy", m1_MPB.neff)
    np.save("test_data/stripWG_mpb/E1.npy", m1_MPB.E)
    np.save("test_data/stripWG_mpb/H1.npy", m1_MPB.H)
    np.save("test_data/stripWG_mpb/y1.npy", m1_MPB.y)
    np.save("test_data/stripWG_mpb/z1.npy", m1_MPB.z)
    np.save("test_data/stripWG_mpb/neff2.npy", m2_MPB.neff)
    np.save("test_data/stripWG_mpb/E2.npy", m2_MPB.E)
    np.save("test_data/stripWG_mpb/H2.npy", m2_MPB.H)
    np.save("test_data/stripWG_mpb/y2.npy", m2_MPB.y)
    np.save("test_data/stripWG_mpb/z2.npy", m2_MPB.z)


def compare_mpb_lumerical(plot=False) -> None:
    """
    WARNING: Segmentation fault occurs if both ms object above and sim object exist in memory at the same time
    Instead load results from separate MPB run

    Same namespace run does not work
    # MPB mode
    # ms = get_mode_solver_rib(wg_width=0.5)
    # modes = find_modes_waveguide(mode_solver=ms, res=50)
    # m1_MPB = modes[1]

    separate namespace run does not work either
    # m1_MPB = MPB_eigenmode()
    """
    # Test data
    filepath = gf.CONFIG["module_path"] / "simulation" / "gmeep" / "test_data"

    # MPB calculation
    # Load previously-computed waveguide results
    m1_MPB_neff = np.load(filepath / "stripWG_mpb" / "neff1.npy")
    m1_MPB_E = np.load(filepath / "stripWG_mpb" / "E1.npy")
    m1_MPB_H = np.load(filepath / "stripWG_mpb" / "H1.npy")
    m1_MPB_y = np.load(filepath / "stripWG_mpb" / "y1.npy")
    m1_MPB_z = np.load(filepath / "stripWG_mpb" / "z1.npy")
    m2_MPB_neff = np.load(filepath / "stripWG_mpb" / "neff2.npy")
    m2_MPB_E = np.load(filepath / "stripWG_mpb" / "E2.npy")
    m2_MPB_H = np.load(filepath / "stripWG_mpb" / "H2.npy")
    m2_MPB_y = np.load(filepath / "stripWG_mpb" / "y2.npy")
    m2_MPB_z = np.load(filepath / "stripWG_mpb" / "z2.npy")
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
    m2_MPB = Mode(
        mode_number=1,
        neff=m2_MPB_neff,
        wavelength=None,
        ng=None,
        E=m2_MPB_E,
        H=m2_MPB_H,
        eps=None,
        y=m2_MPB_y,
        z=m2_MPB_z,
    )

    # Load Lumerical result
    with h5py.File(filepath / "stripWG_lumerical" / "mode1.mat", "r") as f:
        E, H, y, z = lumerical_parser(
            f["E"]["E"], f["H"]["H"], f["E"]["y"], f["E"]["z"], res=50
        )
        # Package into modes object
        m1_lumerical = Mode(
            mode_number=1,
            neff=f["neff"][0][0][0],
            wavelength=None,
            ng=None,
            E=E,
            H=H,
            eps=None,
            y=y,
            z=z,
        )
    with h5py.File(filepath / "stripWG_lumerical" / "mode2.mat", "r") as f:
        E, H, y, z = lumerical_parser(
            f["E"]["E"], f["H"]["H"], f["E"]["y"], f["E"]["z"], res=50
        )
        # Package into modes object
        m2_lumerical = Mode(
            mode_number=1,
            neff=f["neff"][0][0][0],
            wavelength=None,
            ng=None,
            E=E,
            H=H,
            eps=None,
            y=y,
            z=z,
        )

    # MEEP calculation
    c = straight(length=2, width=0.45)
    c = c.copy()
    c = c.add_padding(default=0, bottom=4, top=4, layers=[(100, 0)])

    sim_dict = get_simulation(
        c,
        is_3d=True,
        port_source_offset=-0.1,
        port_monitor_offset=-0.1,
        port_margin=3,
        resolution=50,
    )

    m1_MEEP = get_port_2Dx_eigenmode(
        sim_dict=sim_dict,
        source_index=0,
        port_name="o1",
    )

    m2_MEEP = get_port_2Dx_eigenmode(
        sim_dict=sim_dict,
        source_index=0,
        port_name="o1",
        band_num=2,
    )

    if plot:
        # M1, E-field
        plt.figure(figsize=(10, 8), dpi=100)
        plt.suptitle(
            "MEEP get_eigenmode / MPB find_modes_waveguide / Lumerical (manual)",
            y=1.05,
            fontsize=18,
        )

        plt.subplot(3, 3, 1)
        m1_MEEP.plot_ex(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 2)
        m1_MPB.plot_ex(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 3)
        m1_lumerical.plot_ex(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 4)
        m1_MEEP.plot_ey(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 5)
        m1_MPB.plot_ey(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 6)
        m1_lumerical.plot_ey(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 7)
        m1_MEEP.plot_ez(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 8)
        m1_MPB.plot_ez(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 9)
        m1_lumerical.plot_ez(show=False, operation=np.abs, scale=False)

        plt.tight_layout()
        plt.show()

        # M1, H-field
        plt.figure(figsize=(10, 8), dpi=100)
        plt.suptitle(
            "MEEP get_eigenmode / MPB find_modes_waveguide / Lumerical (manual)",
            y=1.05,
            fontsize=18,
        )

        plt.subplot(3, 3, 1)
        m1_MEEP.plot_hx(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 2)
        m1_MPB.plot_hx(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 3)
        m1_lumerical.plot_hx(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 4)
        m1_MEEP.plot_hy(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 5)
        m1_MPB.plot_hy(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 6)
        m1_lumerical.plot_hy(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 7)
        m1_MEEP.plot_hz(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 8)
        m1_MPB.plot_hz(show=False, operation=np.abs, scale=False)

        plt.subplot(3, 3, 9)
        m1_lumerical.plot_hz(show=False, operation=np.abs, scale=False)

        plt.tight_layout()
        plt.show()

        # # M2, E-field
        # plt.figure(figsize=(10, 8), dpi=100)

        # plt.subplot(3, 3, 1)
        # m2_MEEP.plot_ex(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 2)
        # m2_MPB.plot_ex(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 3)
        # m2_lumerical.plot_ex(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 4)
        # m2_MEEP.plot_ey(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 5)
        # m2_MPB.plot_ey(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 6)
        # m2_lumerical.plot_ey(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 7)
        # m2_MEEP.plot_ez(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 8)
        # m2_MPB.plot_ez(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 9)
        # m2_lumerical.plot_ez(show=False, operation=np.abs, scale=False)

        # plt.tight_layout()
        # plt.show()

        # # M2, H-field
        # plt.figure(figsize=(10, 8), dpi=100)

        # plt.subplot(3, 3, 1)
        # m2_MEEP.plot_hx(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 2)
        # m2_MPB.plot_hx(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 3)
        # m2_lumerical.plot_hx(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 4)
        # m2_MEEP.plot_hy(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 5)
        # m2_MPB.plot_hy(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 6)
        # m2_lumerical.plot_hy(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 7)
        # m2_MEEP.plot_hz(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 8)
        # m2_MPB.plot_hz(show=False, operation=np.abs, scale=False)

        # plt.subplot(3, 3, 9)
        # m2_lumerical.plot_hz(show=False, operation=np.abs, scale=False)

        # plt.tight_layout()
        # plt.show()

    # Check propagation constants
    # print(m1_MEEP.neff, m1_MPB.neff, m1_lumerical.neff)
    # print(m2_MEEP.neff, m2_MPB.neff, m2_lumerical.neff)

    # Check mode profiles
    assert np.isclose(m1_MPB.neff, m1_lumerical.neff, atol=0.02)
    assert np.isclose(m1_MEEP.neff, m1_MPB.neff, atol=0.02)
    assert np.isclose(m2_MPB.neff, m2_lumerical.neff, atol=0.07)
    assert np.isclose(m2_MEEP.neff, m2_MPB.neff, atol=0.07)

    # TODO modes check


if __name__ == "__main__":
    # MPB_eigenmode_toDisk()
    compare_mpb_lumerical(plot=False)
