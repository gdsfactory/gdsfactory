from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from meep import mpb
from pydantic import BaseModel
from scipy.interpolate import RectBivariateSpline

from gdsfactory.typings import Array

# cmap_default = 'viridis'
cmap_default = "RdBu"


class Mode(BaseModel):
    """Mode object.

    Args:
        mode_number: for the mode.
        wavelength: um for the mode.
        neff: effective index.
        ng: group index.
        fraction_te: from 0 to 1.
        fraction_tm: from 0 to 1.
        effective_area: in um2.
        E: field.
        H: field.
        eps: permittivity.
        y: width.
        z: thickness.

    """

    mode_number: int
    wavelength: float
    neff: float
    ng: Optional[float] = None
    fraction_te: Optional[float] = None
    fraction_tm: Optional[float] = None
    effective_area: Optional[float] = None
    E: Optional[Array[float]] = None
    H: Optional[Array[float]] = None
    eps: Optional[Array[float]] = None
    y: Optional[Array[float]] = None
    z: Optional[Array[float]] = None

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"Mode{self.mode_number}"

    def E_grid_interp(self, y_arr, z_arr, index):
        """Creates new attributes with scipy.interpolate.RectBivariateSpline objects that can be used to interpolate the field on a new regular grid.

        Args:
            y_grid (np.array): y values where to evaluate, in increasing array.
            z_grid (np.array): z values where to evaluate, in increasing array.
            index: 0: x, 1: y, 2: z

        """
        if index not in [0, 1, 2]:
            raise ValueError(f"index = {index} needs to be (0: x, 1: y, 2: z)")

        return np.flip(
            RectBivariateSpline(self.y, self.z, np.real(self.E[:, :, 0, index]))(
                y_arr, z_arr, grid=True
            )
            + (
                1j
                * RectBivariateSpline(self.y, self.z, np.imag(self.E[:, :, 0, index]))(
                    y_arr, z_arr, grid=True
                )
            )
        )

    def Ex_grid_interp(self, y_arr, z_arr):
        return self.E_grid_interp(y_arr=y_arr, z_arr=z_arr, index=0)

    def Ey_grid_interp(self, y_arr, z_arr):
        return self.E_grid_interp(y_arr=y_arr, z_arr=z_arr, index=1)

    def Ez_grid_interp(self, y_arr, z_arr):
        return self.E_grid_interp(y_arr=y_arr, z_arr=z_arr, index=2)

    def H_grid_interp(self, y_arr, z_arr, index=0):
        """Creates new attributes with scipy.interpolate.RectBivariateSpline objects that can be used to interpolate the field on a new regular grid.

        Args:
            y_grid (np.array): y values where to evaluate, in increasing array.
            z_grid (np.array): z values where to evaluate, in increasing array.
            index: 0: x, 1: y, 2: z

        """
        if index not in [0, 1, 2]:
            raise ValueError(f"index = {index} needs to be (0: x, 1: y, 2: z)")
        return np.flip(
            RectBivariateSpline(self.y, self.z, np.real(self.H[:, :, 0, index]))(
                y_arr, z_arr, grid=True
            )
            + (
                1j
                * RectBivariateSpline(self.y, self.z, np.imag(self.H[:, :, 0, index]))(
                    y_arr, z_arr, grid=True
                )
            )
        )

    def Hx_grid_interp(self, y_arr, z_arr):
        return self.H_grid_interp(y_arr=y_arr, z_arr=z_arr, index=0)

    def Hy_grid_interp(self, y_arr, z_arr):
        return self.H_grid_interp(y_arr=y_arr, z_arr=z_arr, index=1)

    def Hz_grid_interp(self, y_arr, z_arr):
        return self.H_grid_interp(y_arr=y_arr, z_arr=z_arr, index=2)

    def plot_eps(
        self,
        cmap: str = "binary",
        origin="lower",
        logscale: bool = False,
        show: bool = True,
    ) -> None:
        """Plot index profile."""
        plt.imshow(
            self.eps**0.5,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
        )
        plt.title("index profile")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_e(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
    ) -> None:
        """Plot Electric field module."""
        E = self.E / abs(max(self.E.min(), self.E.max(), key=abs)) if scale else self.E
        Eabs = np.sqrt(
            np.multiply(E[:, :, 0, 2], E[:, :, 0, 2])
            + np.multiply(E[:, :, 0, 1], E[:, :, 0, 1])
            + np.multiply(E[:, :, 0, 0], E[:, :, 0, 0])
        )
        ep = abs(Eabs)
        ep = 10 * np.log10(ep) if logscale else ep
        plt.imshow(
            ep.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=0 if scale else ep.min(),
            vmax=1 if scale else ep.max(),
        )
        plt.title("$|E|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_ex(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ) -> None:
        E = self.E / abs(max(self.E.min(), self.E.max(), key=abs)) if scale else self.E
        ex = E[:, :, 0, 0]
        ex = 10 * np.log10(np.abs(ex)) if logscale else operation(ex)
        plt.imshow(
            ex.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=-1 if scale else ex.min(),
            vmax=1 if scale else ex.max(),
        )
        plt.title("$E_x$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_ey(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ) -> None:
        E = self.E / abs(max(self.E.min(), self.E.max(), key=abs)) if scale else self.E
        ey = E[:, :, 0, 1]
        ey = 10 * np.log10(np.abs(ey)) if logscale else operation(ey)
        plt.imshow(
            ey.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=-1 if scale else ey.min(),
            vmax=1 if scale else ey.max(),
        )
        plt.title("$E_y$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_ez(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ) -> None:
        E = self.E / abs(max(self.E.min(), self.E.max(), key=abs)) if scale else self.E
        ez = E[:, :, 0, 2]
        ez = 10 * np.log10(ez) if logscale else operation(ez)
        plt.imshow(
            ez.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=-1 if scale else ez.min(),
            vmax=1 if scale else ez.max(),
        )
        plt.title("$E_z$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_e_all(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ) -> None:
        plt.figure(figsize=(16, 10), dpi=100)

        plt.subplot(2, 3, 1)
        self.plot_ex(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 2)
        self.plot_ey(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 3)
        self.plot_ez(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 4)
        self.plot_e(show=False, scale=scale)

        plt.subplot(2, 3, 5)
        self.plot_eps(show=False)

        plt.tight_layout()
        plt.show()

    def plot_h(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
    ) -> None:
        H = self.H / abs(max(self.H.min(), self.H.max(), key=abs)) if scale else self.H
        Habs = np.sqrt(
            np.multiply(H[:, :, 0, 2], H[:, :, 0, 2])
            + np.multiply(H[:, :, 0, 1], H[:, :, 0, 1])
            + np.multiply(H[:, :, 0, 0], H[:, :, 0, 0])
        )
        hp = abs(Habs)
        hp = 10 * np.log10(hp) if logscale else hp
        plt.imshow(
            hp.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=0 if scale else hp.min(),
            vmax=1 if scale else hp.max(),
        )
        plt.title("$|H|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_hx(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ) -> None:
        H = self.H / abs(max(self.H.min(), self.H.max(), key=abs)) if scale else self.H
        hx = H[:, :, 0, 0]
        hx = 10 * np.log10(np.abs(hx)) if logscale else operation(hx)
        plt.imshow(
            hx.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=-1 if scale else hx.min(),
            vmax=1 if scale else hx.max(),
        )
        plt.title("$H_x$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_hy(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ) -> None:
        H = self.H / abs(max(self.H.min(), self.H.max(), key=abs)) if scale else self.H
        hy = H[:, :, 0, 1]
        hy = 10 * np.log10(np.abs(hy)) if logscale else operation(hy)
        plt.imshow(
            hy.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=-1 if scale else hy.min(),
            vmax=1 if scale else hy.max(),
        )
        plt.title("$H_y$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_hz(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ) -> None:
        H = self.H / abs(max(self.H.min(), self.H.max(), key=abs)) if scale else self.H
        hz = abs(H[:, :, 0, 2])
        hz = 10 * np.log10(hz) if logscale else operation(hz)
        plt.imshow(
            hz.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=0 if scale else hz.min(),
            vmax=1 if scale else hz.max(),
        )
        plt.title("$H_z$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_h_all(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ) -> None:
        plt.figure(figsize=(16, 10), dpi=100)

        plt.subplot(2, 3, 1)
        self.plot_hx(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 2)
        self.plot_hy(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 3)
        self.plot_hz(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 4)
        self.plot_h(show=False, scale=scale)

        plt.subplot(2, 3, 5)
        self.plot_eps(show=False)

        plt.tight_layout()
        plt.show()


class Waveguide(BaseModel):
    """Waveguide Model.

    Args:
        wg_width: um
        wg_thickness: um
        slab_thickness: um
        ncore: core refractive index.
        nclad: cladding refractive index.
        sy: float
        sz: float
        resolution: int
        nmodes: int
        modes: Dict[Mode]

    """

    wg_width: float = 0.45
    wg_thickness: float = 0.22
    slab_thickness: float = 0
    ncore: float = 3.47
    nclad: float = 1.44
    sy: float = 2.0
    sz: float = 2.0
    resolution: int = 32
    nmodes: int = 4
    modes: Optional[Dict[int, Mode]] = None


class WavelengthSweep(BaseModel):
    wavelength: List[float]
    neff: Dict[int, List[float]]
    ng: Dict[int, List[float]]


ModeSolverFactory = Callable[..., mpb.ModeSolver]
ModeSolverOrFactory = Union[mpb.ModeSolver, ModeSolverFactory]


if __name__ == "__main__":
    import gdsfactory.simulation.modes as gm

    m = gm.find_modes_waveguide()
    # m[1].plot_e_all(operation=np.abs)
    # plt.show()
    # m[1].plot_h_all(operation=np.abs)
    # w = Waveguide()

    ys = np.linspace(-3, 3, 2000)
    zs = np.linspace(-1.5, 1.5, 2000)
    m[1].grid_interp()

    Ex_interp = m[1].Ex_grid_interp(ys, zs)
    Ey_interp = m[1].Ey_grid_interp(ys, zs)
    Ez_interp = m[1].Ez_grid_interp(ys, zs)

    plt.figure(figsize=(10, 8), dpi=100)

    plt.subplot(3, 2, 1)
    plt.title("As calculated")
    m[1].plot_ex(show=False, operation=np.abs, scale=False)

    plt.subplot(3, 2, 2)
    plt.title("Interp")
    plt.imshow(
        np.abs(Ex_interp).T,
        aspect="auto",
        extent=[np.min(ys), np.max(ys), np.min(zs), np.max(zs)],
    )
    plt.colorbar()

    plt.subplot(3, 2, 3)
    plt.title("As calculated")
    m[1].plot_ey(show=False, operation=np.abs, scale=False)

    plt.subplot(3, 2, 4)
    plt.title("Interp")
    plt.imshow(
        np.abs(Ey_interp).T,
        aspect="auto",
        extent=[np.min(ys), np.max(ys), np.min(zs), np.max(zs)],
    )
    plt.colorbar()

    plt.subplot(3, 2, 5)
    plt.title("As calculated")
    m[1].plot_ez(show=False, operation=np.abs, scale=False)

    plt.subplot(3, 2, 6)
    plt.title("Interp")
    plt.imshow(
        np.abs(Ez_interp).T,
        aspect="auto",
        extent=[np.min(ys), np.max(ys), np.min(zs), np.max(zs)],
    )
    plt.colorbar()

    plt.show()
