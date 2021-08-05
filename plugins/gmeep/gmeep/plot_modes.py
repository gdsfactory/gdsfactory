"""Plot modes.
"""

import pathlib
from pathlib import PosixPath
from typing import Optional, Union

import matplotlib.pyplot as plt
import meep as mp
from meep import mpb
import numpy as np

mp.verbosity(0)
mp.verbosity.mpb = 0


def plot_modes(
    mode_solver: mpb.ModeSolver,
    sx: float = 2.0,
    sy: float = 2.0,
    mode_number: int = 1,
    logscale: bool = False,
    plotH: bool = False,
    dirpath: Optional[Union[PosixPath, str]] = None,
    polarization: str = "TE",
    cmap_fields: str = "viridis",
    cmap_geom: str = "viridis",
):
    """Plot mode fields.

    Args:
        sx: Size of the simulation region in the x-direction (um) (default=4.0)
        sy: Size of the simulation region in the y-direction (um) (default=4.0)
        mode_number: Which mode to plot (only plots one mode at a time).  Must be a number equal to or less than num_mode (default=1)
        plotH: plot magnetic field.
        dirpath: to save fields waveguide cross-sections fields (top-down and side-view) if savefig=True
        cmap_geom: colormap for geometry
        cmap_fields: colormap for fields (hot_r, coolwarm, viridis)
    """

    origin = "upper"

    # plot electric field
    eps = mode_solver.get_epsilon()
    mode_solver.get_dfield(mode_number)
    E = mode_solver.get_efield(mode_number)
    Eabs = np.sqrt(
        np.multiply(E[:, :, 0, 2], E[:, :, 0, 2])
        + np.multiply(E[:, :, 0, 1], E[:, :, 0, 1])
        + np.multiply(E[:, :, 0, 0], E[:, :, 0, 0])
    )
    H = mode_solver.get_hfield(mode_number)
    Habs = np.sqrt(
        np.multiply(H[:, :, 0, 2], H[:, :, 0, 2])
        + np.multiply(H[:, :, 0, 1], H[:, :, 0, 1])
        + np.multiply(H[:, :, 0, 0], H[:, :, 0, 0])
    )

    plt_extent = [-sy / 2.0, +sy / 2.0, -sx / 2.0, +sx / 2.0]

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    ex = abs(E[:, :, 0, 2])
    ex = 10 * np.log10(ex) if logscale else ex
    plt.imshow(
        ex.T,
        cmap=cmap_fields,
        origin=origin,
        aspect="auto",
        extent=plt_extent,
    )
    plt.title("Waveguide mode $|E_x|$")
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
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
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
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
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
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
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(
        eps.T ** 0.5, cmap=cmap_geom, origin=origin, aspect="auto", extent=plt_extent
    )
    plt.title("index profile")
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
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
        plt.ylabel("y-axis")
        plt.xlabel("x-axis")
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
        plt.ylabel("y-axis")
        plt.xlabel("x-axis")
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
        plt.ylabel("y-axis")
        plt.xlabel("x-axis")
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
        plt.ylabel("y-axis")
        plt.xlabel("x-axis")
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
        plt.xlabel("x-axis")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

        if dirpath:
            plt.savefig(dirpath / f"{polarization}_mode{mode_number}_H.png")


if __name__ == "__main__":
    from gmeep.find_modes import find_modes

    r = find_modes()
    plot_modes(mode_solver=r["mode_solver"])
