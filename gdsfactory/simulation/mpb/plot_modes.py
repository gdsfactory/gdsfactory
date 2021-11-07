"""Plot modes profiles."""

import pathlib
from pathlib import PosixPath
from typing import Optional, Union

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from meep import mpb

from gdsfactory.simulation.mpb.disable_print import disable_print, enable_print
from gdsfactory.simulation.mpb.get_mode_solver_rib import get_mode_solver_rib
from gdsfactory.simulation.mpb.types import ModeSolverOrFactory

mpb.verbosity(0)
mp.verbosity.mpb = 0


def plot_modes(
    mode_solver: ModeSolverOrFactory = get_mode_solver_rib,
    sy: float = 2.0,
    sz: float = 2.0,
    logscale: bool = False,
    plotH: bool = False,
    dirpath: Optional[Union[PosixPath, str]] = None,
    parity=mp.NO_PARITY,
    polarization: str = "TE",
    cmap_fields: str = "viridis",
    cmap_geom: str = "viridis",
    mode_number: int = 1,
    wavelength: float = 1.55,
    tol: float = 1e-6,
    **kwargs,
):
    """Plot mode fields.

    Args:
        mode_solver: function to get a mode solver or mode_solver
        sy: Size of the simulation region in the y-direction (um)
        sz: Size of the simulation region in the z-direction (um)
        mode_number: to plot. Must be equal to or less than nmodes
        plotH: plot magnetic field.
        dirpath: to save fields
        cmap_fields: colormap for fields (hot_r, coolwarm, viridis)
        cmap_geom: colormap for geometry
    """
    mode_solver = mode_solver(**kwargs) if callable(mode_solver) else mode_solver
    nmodes = mode_solver.nmodes

    omega = 1 / wavelength
    disable_print()
    mode_solver.find_k(
        parity,
        omega,
        mode_number,
        mode_number + nmodes,
        mp.Vector3(1),
        tol,
        omega * 2.02,
        omega * 0.01,
        omega * 10,
        mpb.output_poynting_x,
        mpb.display_yparities,
        mpb.display_group_velocities,
    )
    enable_print()

    E = mode_solver.get_efield(mode_number)
    H = mode_solver.get_hfield(mode_number)
    eps = mode_solver.get_epsilon()
    mode_solver.get_dfield(mode_number)
    Eabs = np.sqrt(
        np.multiply(E[:, :, 0, 2], E[:, :, 0, 2])
        + np.multiply(E[:, :, 0, 1], E[:, :, 0, 1])
        + np.multiply(E[:, :, 0, 0], E[:, :, 0, 0])
    )
    Habs = np.sqrt(
        np.multiply(H[:, :, 0, 2], H[:, :, 0, 2])
        + np.multiply(H[:, :, 0, 1], H[:, :, 0, 1])
        + np.multiply(H[:, :, 0, 0], H[:, :, 0, 0])
    )

    plt_extent = [-sy / 2.0, +sy / 2.0, -sz / 2.0, +sz / 2.0]

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 3, 1)
    ex = abs(E[:, :, 0, 2])
    ex = 10 * np.log10(ex) if logscale else ex
    origin = "upper"
    plt.imshow(
        ex.T,
        cmap=cmap_fields,
        origin=origin,
        aspect="auto",
        extent=plt_extent,
    )
    plt.title("Waveguide mode $|E_x|$")
    plt.ylabel("z-axis")
    plt.xlabel("y-axis")
    plt.colorbar()

    plt.subplot(2, 3, 2)
    ey = abs(E[:, :, 0, 1])
    ey = 10 * np.log10(ey) if logscale else ey
    plt.imshow(
        ey.T,
        cmap=cmap_fields,
        origin=origin,
        aspect="auto",
        extent=plt_extent,
    )
    plt.title("Waveguide mode $|E_y|$")
    plt.ylabel("z-axis")
    plt.xlabel("y-axis")
    plt.colorbar()

    plt.subplot(2, 3, 3)
    ez = abs(E[:, :, 0, 0])
    ez = 10 * np.log10(ez) if logscale else ez
    plt.imshow(
        ez.T,
        cmap=cmap_fields,
        origin=origin,
        aspect="auto",
        extent=plt_extent,
    )
    plt.title("Waveguide mode $|E_z|$")
    plt.ylabel("z-axis")
    plt.xlabel("y-axis")
    plt.colorbar()

    plt.subplot(2, 3, 4)
    ep = abs(Eabs)
    ep = 10 * np.log10(ep) if logscale else ep
    plt.imshow(
        ep.T,
        cmap=cmap_fields,
        origin=origin,
        aspect="auto",
        extent=plt_extent,
    )
    plt.title("Waveguide mode $|E|$")
    plt.ylabel("z-axis")
    plt.xlabel("y-axis")
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(
        eps.T ** 0.5, cmap=cmap_geom, origin=origin, aspect="auto", extent=plt_extent
    )
    plt.title("index profile")
    plt.ylabel("z-axis")
    plt.xlabel("y-axis")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    if dirpath:
        dirpath = pathlib.Path(dirpath)
        dirpath.mkdir(exist_ok=True, parents=True)
        plt.savefig(dirpath / f"{polarization}_mode{mode_number}_E.png")

    if plotH:
        # plot magnetic field
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 3, 1)
        hx = abs(H[:, :, 0, 2])
        hx = 10 * np.log10(hx) if logscale else hx
        plt.imshow(
            hx,
            cmap=cmap_fields,
            origin=origin,
            aspect="auto",
            extent=plt_extent,
        )
        plt.title("Waveguide mode $|H_x|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()

        plt.subplot(2, 3, 2)
        hy = abs(H[:, :, 0, 1])
        hy = 10 * np.log10(hy) if logscale else hy
        plt.imshow(
            hy,
            cmap=cmap_fields,
            origin=origin,
            aspect="auto",
            extent=plt_extent,
        )
        plt.title("Waveguide mode $|H_y|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()

        plt.subplot(2, 3, 3)
        hz = abs(H[:, :, 0, 0])
        hz = 10 * np.log10(hz) if logscale else hz
        plt.imshow(
            hz,
            cmap=cmap_fields,
            origin=origin,
            aspect="auto",
            extent=plt_extent,
        )
        plt.title("Waveguide mode $|H_z|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()

        plt.subplot(2, 3, 4)
        hp = abs(Habs)
        plt.imshow(
            hp,
            cmap=cmap_fields,
            origin=origin,
            aspect="auto",
            extent=plt_extent,
        )
        plt.title("Waveguide mode $|H|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()

        plt.subplot(2, 3, 5)
        plt.imshow(
            eps ** 0.5,
            cmap=cmap_geom,
            origin=origin,
            aspect="auto",
            extent=plt_extent,
        )
        plt.title("index profile")
        plt.ylabel("y-axis")
        plt.xlabel("z-axis")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

        if dirpath:
            plt.savefig(dirpath / f"{polarization}_mode{mode_number}_H.png")


if __name__ == "__main__":
    plot_modes()
    plt.show()
