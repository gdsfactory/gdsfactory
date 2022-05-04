"""tidy3d mode solver

tidy3d has a powerful open source mode solver.

tidy3d can:

- compute bend modes.

TODO:

- cache modes in local file
- include mode overlaps for bend loss
- dipersion calculation

Maybe:
- combine modes package, mpb and tidy3d APIs
"""

import pathlib
import pickle
from types import SimpleNamespace
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from pydantic import BaseModel, Extra
from scipy.constants import c as SPEED_OF_LIGHT
from tidy3d.plugins.mode.solver import compute_modes

from gdsfactory.simulation.gtidy3d.materials import si, sio2
from gdsfactory.types import PathType


def plot(
    X,
    Y,
    n,
    mode=None,
    num_levels: int = 8,
    n_cmap=None,
    mode_cmap=None,
    axes="xy",
    title=None,
    normalize_mode=False,
):
    """Plot mode in matplotlib"""
    x, y = axes
    if n_cmap is None:
        n_cmap = colors.LinearSegmentedColormap.from_list(
            name="custom", colors=["#ffffff", "#c1d9ed"]
        )
    if mode_cmap is None:
        mode_cmap = "inferno"
    if mode is not None:
        mode = np.abs(mode)
        if normalize_mode:
            mode = mode / mode.max()
        plt.contour(
            X,
            Y,
            mode,
            cmap=mode_cmap,
            levels=np.linspace(mode.min(), mode.max(), num_levels),
        )
        plt.colorbar(label="mode")
    n = np.real(n)
    plt.pcolormesh(X, Y, n, cmap=n_cmap)
    plt.colorbar(label="n")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.4)
    plt.axis("scaled")
    if title is not None:
        plt.title(title)


def _mesh_1d(x_min, x_max, mesh_x):
    if isinstance(mesh_x, int):
        x = np.linspace(x_min, x_max, mesh_x + 1)
        dx = x[1] - x[0]
    elif isinstance(mesh_x, float):
        dx = mesh_x
        x = np.arange(x_min, x_max + 0.999 * dx, dx)
        mesh_x = x.shape[0]
    else:
        raise ValueError("Invalid 'mesh_x'")
    return x, mesh_x, dx


def create_mesh(x_min, y_min, x_max, y_max, mesh_x, mesh_y):
    x, mesh_x, dx = _mesh_1d(x_min, x_max, mesh_x)
    y, mesh_y, dy = _mesh_1d(y_min, y_max, mesh_y)

    Yx, Xx = np.meshgrid(y[:-1], x[:-1] + dx / 2)
    Yy, Xy = np.meshgrid(y[:-1] + dy / 2, x[:-1])
    Yz, Xz = np.meshgrid(y[:-1], x[:-1])
    return x, y, Xx, Yx, Xy, Yy, Xz, Yz


def get_n(
    Y,
    Z,
    ncore: float,
    nclad: float,
    wg_width: float,
    t_box: float,
    t_wg: float,
    t_slab: float,
    t_clad: float,
):
    """Return index matrix."""
    w = wg_width
    n = np.ones_like(Y)
    n[
        (-w / 2 - t_clad / 2 <= Y)
        & (Y <= w / 2 + t_clad / 2)
        & (Z >= t_box)
        & (Z <= t_box + t_wg + t_clad)
    ] = nclad
    n[(Z <= 1.0 + t_slab + t_clad)] = nclad
    n[(-w / 2 <= Y) & (Y <= w / 2) & (Z >= t_box) & (Z <= t_box + t_wg)] = ncore
    n[(Z >= t_box) & (Z <= t_box + t_slab)] = ncore if t_slab else nclad
    return n


class Waveguide(BaseModel):
    """Waveguide Model.

    Args:
        wavelength: (um).
        wg_width: waveguide width.
        t_wg: thickness waveguide (um).
        t_slab: thickness slab (um).
        t_box: thickness BOX (um).
        t_clad: thickness cladding (um).
        ncore: core refractive index.
        nclad: cladding refractive index.
        w_sim: width simulation (um).
        resolution: pixels/um.
        nmodes: number of modes to compute.
        bend_radius: optional bend radius (um).
    """

    wavelength: float = 1.55
    wg_width: float = 0.45
    t_wg: float = 0.22
    t_slab: float = 0.0
    t_box: float = 2.0
    t_clad: float = 2.0
    ncore: Callable[str, float] = si
    nclad: Callable[str, float] = sio2
    w_sim: float = 2.0
    resolution: int = 100
    nmodes: int = 10
    bend_radius: Optional[float] = None

    class Config:
        extra = Extra.allow

    @property
    def t_sim(self):
        return self.t_box + self.t_wg + self.t_clad

    def plot_index(self, wavelength: Optional[float] = None) -> None:
        wavelength = wavelength or self.wavelength
        x, y, Xx, Yx, Xy, Yy, Xz, Yz = create_mesh(
            -self.w_sim / 2,
            0.0,
            self.w_sim / 2,
            self.t_sim,
            self.resolution,
            self.resolution,
        )

        nx = get_n(
            Xx,
            Yx,
            wg_width=self.wg_width,
            ncore=self.ncore(wavelength),
            nclad=self.nclad(wavelength),
            t_box=self.t_box,
            t_slab=self.t_slab,
            t_wg=self.t_wg,
            t_clad=self.t_clad,
        )
        plot(Xx, Yx, nx)

    def compute_modes(self, wavelength: Optional[float] = None) -> None:
        wavelength = wavelength or self.wavelength
        x, y, Xx, Yx, Xy, Yy, Xz, Yz = create_mesh(
            -self.w_sim / 2,
            0.0,
            self.w_sim / 2,
            self.t_sim,
            self.resolution,
            self.resolution,
        )

        nx = get_n(
            Xx,
            Yx,
            wg_width=self.wg_width,
            ncore=self.ncore(wavelength),
            nclad=self.nclad(wavelength),
            t_box=self.t_box,
            t_slab=self.t_slab,
            t_wg=self.t_wg,
            t_clad=self.t_clad,
        )
        ny = get_n(
            Xy,
            Yy,
            wg_width=self.wg_width,
            ncore=self.ncore(wavelength),
            nclad=self.nclad(wavelength),
            t_box=self.t_box,
            t_slab=self.t_slab,
            t_wg=self.t_wg,
            t_clad=self.t_clad,
        )
        nz = get_n(
            Xz,
            Yz,
            wg_width=self.wg_width,
            ncore=self.ncore(wavelength),
            nclad=self.nclad(wavelength),
            t_box=self.t_box,
            t_slab=self.t_slab,
            t_wg=self.t_wg,
            t_clad=self.t_clad,
        )

        ((Ex, Ey, Ez), (Hx, Hy, Hz)), neffs = (
            x.squeeze()
            for x in compute_modes(
                eps_cross=[nx**2, ny**2, nz**2],
                coords=[x, y],
                freq=SPEED_OF_LIGHT / (wavelength * 1e-6),
                mode_spec=SimpleNamespace(
                    num_modes=10,
                    bend_radius=self.bend_radius,
                    bend_axis=1,
                    angle_theta=0.0,
                    angle_phi=0.0,
                    num_pml=(0, 0),
                    target_neff=self.ncore(wavelength),
                    sort_by="largest_neff",
                ),
            )
        )

        self.nx, self.ny, self.nz = nx, ny, nz
        self.Xx, self.Yx, self.Xy, self.Yy, self.Xz, self.Yz = Xx, Yx, Xy, Yy, Xz, Yz
        self.Ex, self.Ey, self.Ez = Ex, Ey, Ez
        self.Hx, self.Hy, self.Hz = Hx, Hy, Hz
        self.neffs = neffs

    def plot_Ex(self, index: int = 0) -> None:
        if not hasattr(self, "neffs"):
            self.compute_modes()

        nx, neffs, Ex = self.nx, self.neffs, self.Ex
        neff_, Ex_ = np.real(neffs[index]), Ex[..., index]
        plot(self.Xx, self.Yx, nx, mode=np.abs(Ex_) ** 2, title=f"Ex::{neff_:.3f}")
        plt.show()

    def plot_Ey(self, index: int = 0) -> None:
        if not hasattr(self, "neffs"):
            self.compute_modes()

        nx, neffs, Ey = self.nx, self.neffs, self.Ey
        neff_, Ey_ = np.real(neffs[index]), Ey[..., index]
        plot(self.Xx, self.Yx, nx, mode=np.abs(Ey_) ** 2, title=f"Ey::{neff_:.3f}")
        plt.show()

    def _repr_html_(self):
        """Show index in matplotlib for jupyter notebooks."""
        self.plot_index()
        return self.__repr__()

    def pickle_dump(self, filepath: PathType) -> None:
        data = pickle.dumps(self)
        filepath = pathlib.Path(filepath)
        filepath.write_bytes(data)


def pickle_load(filepath: PathType) -> Waveguide:
    filepath = pathlib.Path(filepath)
    return pickle.loads(filepath.read_bytes())


if __name__ == "__main__":
    c = Waveguide(t_slab=90e-3, bend_radius=5)
    # c.plot_index()
    c.plot_Ex(index=0)
    plt.show()
