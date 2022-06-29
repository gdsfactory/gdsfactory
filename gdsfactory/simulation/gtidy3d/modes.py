"""tidy3d mode solver.

tidy3d has a powerful open source mode solver.

tidy3d can:

- compute bend modes.

TODO:

- fix mode overlaps for bend loss
- calculate dispersion

Maybe:

- combine modes package, mpb and tidy3d APIs
"""

import pathlib
from types import SimpleNamespace
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from pydantic import BaseModel, Extra
from scipy.constants import c as SPEED_OF_LIGHT
from tidy3d.plugins.mode.solver import compute_modes
from tqdm.auto import tqdm

from gdsfactory.config import CONFIG, logger
from gdsfactory.serialization import get_hash
from gdsfactory.simulation.gtidy3d.materials import si, sin, sio2
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
) -> None:
    """Plot mode in matplotlib.

    Args:
        X: x array.
        Y: y array.
        n: refractive index.
        mode: mode number.
        num_levels: for plot.
        n_cmap: refractive index color map.
        mode_cmap: mode color map.
        axes: "xy".
        title: for the plot.
        normalize_mode: divide by maximum value.

    """
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
    wg_thickness: float,
    slab_thickness: float,
    t_clad: float,
):
    """Return index matrix."""
    w = wg_width
    n = np.ones_like(Y) * nclad
    n[
        (-w / 2 - t_clad / 2 <= Y)
        & (Y <= w / 2 + t_clad / 2)
        & (Z >= t_box)
        & (Z <= t_box + wg_thickness + t_clad)
    ] = nclad
    n[(Z <= 1.0 + slab_thickness + t_clad)] = nclad
    n[(-w / 2 <= Y) & (Y <= w / 2) & (Z >= t_box) & (Z <= t_box + wg_thickness)] = ncore
    n[(Z >= t_box) & (Z <= t_box + slab_thickness)] = ncore if slab_thickness else nclad
    return n


SETTINGS = [
    "wavelength",
    "wg_width",
    "wg_thickness",
    "ncore",
    "nclad",
    "slab_thickness",
    "t_box",
    "t_clad",
    "xmargin",
    "resolution",
    "nmodes",
    "bend_radius",
]


class Waveguide(BaseModel):
    """Waveguide Model.

    Parameters:
        wavelength: (um).
        wg_width: waveguide width.
        wg_thickness: thickness waveguide (um).
        ncore: core refractive index.
        nclad: cladding refractive index.
        slab_thickness: thickness slab (um).
        t_box: thickness BOX (um).
        t_clad: thickness cladding (um).
        xmargin: margin from waveguide edge to each side (um).
        resolution: pixels/um.
        nmodes: number of modes to compute.
        bend_radius: optional bend radius (um).
        cache: filepath for caching modes.

    ::

          __________________________
          |
          |
          |         width     xmargin
          |     <----------> <------>
          |      ___________   _ _ _
          |     |           |       |
          |_____|  ncore    |_______|
          |                         | wg_thickness
          |slab_thickness    nslab  |
          |_________________________|
          |
          |        nclad
          |__________________________
          <------------------------>
                   w_sim

    """

    wavelength: float
    wg_width: float
    wg_thickness: float
    ncore: Union[float, Callable[[str], float]]
    nclad: Union[float, Callable[[str], float]]
    slab_thickness: float
    t_box: float = 2.0
    t_clad: float = 2.0
    xmargin: float = 1.0
    resolution: int = 100
    nmodes: int = 4
    bend_radius: Optional[float] = None
    cache: Optional[PathType] = CONFIG["modes"]

    class Config:
        extra = Extra.allow

    @property
    def t_sim(self):
        return self.t_box + self.wg_thickness + self.t_clad

    @property
    def w_sim(self):
        return self.wg_width + 2 * self.xmargin

    @property
    def filepath(self) -> pathlib.Path:
        if self.cache is None:
            return
        cache = pathlib.Path(self.cache)
        cache.mkdir(exist_ok=True, parents=True)
        settings = dict(
            wavelength=self.wavelength,
            wg_width=self.wg_width,
            wg_thickness=self.wg_thickness,
            slab_thickness=self.slab_thickness,
            t_box=self.t_box,
            t_clad=self.t_clad,
            ncore=self.ncore,
            nclad=self.nclad,
            xmargin=self.xmargin,
            resolution=self.resolution,
            nmodes=self.nmodes,
            bend_radius=self.bend_radius,
        )
        return cache / f"{get_hash(settings)}.npz"

    def get_ncore(self, wavelength):
        return self.ncore(wavelength) if callable(self.ncore) else self.ncore

    def get_nclad(self, wavelength):
        return self.nclad(wavelength) if callable(self.nclad) else self.nclad

    def plot_index(self, wavelength: Optional[float] = None) -> None:
        wavelength = wavelength or self.wavelength
        x, y, Xx, Yx, Xy, Yy, Xz, Yz = create_mesh(
            -self.w_sim / 2,
            0.0,
            +self.w_sim / 2,
            self.t_sim,
            self.resolution,
            self.resolution,
        )

        nx = get_n(
            Xx,
            Yx,
            wg_width=self.wg_width,
            ncore=self.get_ncore(wavelength),
            nclad=self.get_nclad(wavelength),
            t_box=self.t_box,
            slab_thickness=self.slab_thickness,
            wg_thickness=self.wg_thickness,
            t_clad=self.t_clad,
        )
        plot(Xx, Yx, nx)
        plt.show()

    def compute_modes(
        self,
        overwrite: bool = False,
    ) -> None:
        if hasattr(self, "neffs") and not overwrite:
            return

        wavelength = self.wavelength
        x, y, Xx, Yx, Xy, Yy, Xz, Yz = create_mesh(
            -self.w_sim / 2,
            0.0,
            +self.w_sim / 2,
            self.t_sim,
            self.resolution,
            self.resolution,
        )

        nx = get_n(
            Xx,
            Yx,
            wg_width=self.wg_width,
            ncore=self.get_ncore(wavelength),
            nclad=self.get_nclad(wavelength),
            t_box=self.t_box,
            slab_thickness=self.slab_thickness,
            wg_thickness=self.wg_thickness,
            t_clad=self.t_clad,
        )
        ny = get_n(
            Xy,
            Yy,
            wg_width=self.wg_width,
            ncore=self.get_ncore(wavelength),
            nclad=self.get_nclad(wavelength),
            t_box=self.t_box,
            slab_thickness=self.slab_thickness,
            wg_thickness=self.wg_thickness,
            t_clad=self.t_clad,
        )
        nz = get_n(
            Xz,
            Yz,
            wg_width=self.wg_width,
            ncore=self.get_ncore(wavelength),
            nclad=self.get_nclad(wavelength),
            t_box=self.t_box,
            slab_thickness=self.slab_thickness,
            wg_thickness=self.wg_thickness,
            t_clad=self.t_clad,
        )
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Xx, self.Yx, self.Xy, self.Yy, self.Xz, self.Yz = Xx, Yx, Xy, Yy, Xz, Yz

        if self.cache and self.filepath.exists():
            data = np.load(self.filepath)
            self.Ex = data["Ex"]
            self.Ey = data["Ey"]
            self.Ez = data["Ez"]
            self.Hx = data["Hx"]
            self.Hy = data["Hy"]
            self.Hz = data["Hz"]
            self.neffs = data["neffs"]
            logger.info(f"load {self.filepath} mode data from file cache.")
            return

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
                    target_neff=self.get_ncore(wavelength),
                    sort_by="largest_neff",
                ),
            )
        )

        self.Ex, self.Ey, self.Ez = Ex, Ey, Ez
        self.Hx, self.Hy, self.Hz = Hx, Hy, Hz
        self.neffs = neffs

        data = dict(
            Ex=self.Ex,
            Ey=self.Ey,
            Ez=self.Ez,
            Hx=self.Hx,
            Hy=self.Hy,
            Hz=self.Hz,
            neffs=self.neffs,
        )
        np.savez_compressed(self.filepath, **data)
        logger.info(f"write {self.filepath} mode data to file cache.")

    def plot_Ex(self, mode_index: int = 0) -> None:
        if not hasattr(self, "neffs"):
            self.compute_modes()

        nx, neffs, Ex = self.nx, self.neffs, self.Ex
        neff_, Ex_ = np.real(neffs[mode_index]), Ex[..., mode_index]
        plot(self.Xx, self.Yx, nx, mode=np.abs(Ex_) ** 2, title=f"Ex::{neff_:.3f}")
        plt.show()

    def plot_Ey(self, mode_index: int = 0) -> None:
        if not hasattr(self, "neffs"):
            self.compute_modes()

        nx, neffs, Ey = self.nx, self.neffs, self.Ey
        neff_, Ey_ = np.real(neffs[mode_index]), Ey[..., mode_index]
        plot(self.Xx, self.Yx, nx, mode=np.abs(Ey_) ** 2, title=f"Ey::{neff_:.3f}")
        plt.show()

    def _repr_html_(self) -> str:
        """Show index in matplotlib for jupyter notebooks."""
        self.plot_index()
        return self.__repr__()

    def __repr__(self) -> str:
        """Show index in matplotlib for jupyter notebooks."""
        return ", ".join([f"{k}:{getattr(self, k)!r}" for k in SETTINGS])

    def get_overlap(
        self, wg: "Waveguide", mode_index1: int = 0, mode_index2: int = 0
    ) -> float:
        """Returns mode overlap integral.

        Args:
            wg: other waveguide.
        """
        wg1 = self
        wg2 = wg

        wg1.compute_modes()
        wg2.compute_modes()

        return np.sum(
            np.conj(wg1.Ex[..., mode_index1]) * wg2.Hy[..., mode_index2]
            - np.conj(wg1.Ey[..., mode_index1]) * wg2.Hx[..., mode_index2]
            + wg2.Ex[..., mode_index2] * np.conj(wg1.Hy[..., mode_index1])
            - wg2.Ey[..., mode_index2] * np.conj(wg1.Hx[..., mode_index1])
        )


def sweep_bend_loss(
    bend_radius_min: float = 2.0,
    bend_radius_max: float = 5,
    steps: int = 4,
    mode_index: int = 0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns overlap integral loss.

    The loss is squared because you hit the bend loss twice
    (from bend to straight and from straight to bend).

    Args:
        bend_radius_min: min bend radius (um).
        bend_radius_max: max bend radius (um).
        steps: number of steps.
        mode_index: where 0 is the fundamental mode.

    Keyword Args:
        wavelength: (um).
        wg_width: waveguide width.
        wg_thickness: thickness waveguide (um).
        slab_thickness: thickness slab (um).
        t_box: thickness BOX (um).
        t_clad: thickness cladding (um).
        ncore: core refractive index.
        nclad: cladding refractive index.
        xmargin: margin from waveguide edge to each side (um).
        resolution: pixels/um.
        nmodes: number of modes to compute.
    """

    r = np.linspace(bend_radius_min, bend_radius_max, steps)
    integral = np.zeros_like(r)

    wg = Waveguide(**kwargs)

    for i, radius in tqdm(enumerate(r)):
        wg_bent = Waveguide(bend_radius=radius, **kwargs)
        wg.get_overlap(wg_bent, mode_index1=mode_index, mode_index2=mode_index)

        # normalized overlap integral
        integral[i] = np.abs(
            wg.get_overlap(wg_bent, mode_index, mode_index) ** 2
            / wg.get_overlap(wg, mode_index, mode_index)
            / wg.get_overlap(wg_bent, mode_index, mode_index)
        )

    return r, integral**2


__all__ = ("sweep_bend_loss", "Waveguide", "si", "sio2", "sin")

if __name__ == "__main__":
    c = Waveguide(
        wavelength=1.55,
        wg_width=0.5,
        wg_thickness=0.22,
        slab_thickness=0.0,
        ncore=si,
        nclad=sio2,
    )
    c.plot_Ex(0)

    # nitride = find_modes(wavelength=1.55, wg_width=1.0, wg_thickness=0.4, ncore=2.0)
    # nitride.plot_index()

    # c = pickle_load("strip.pkl")

    # c0 = Waveguide(slab_thickness=0)
    # c0.plot_Ex(index=0)
    # c0.pickle_dump("strip.pkl")

    # c1 = Waveguide(slab_thickness=0, bend_radius=5)
    # c1.plot_Ex()
    # c1.pickle_dump("strip_bend5.pkl")

    # c = Waveguide(slab_thickness=90e-3, bend_radius=5)
    # c.plot_index()

    # r, integral = sweep_bend_loss(
    #     wavelength=1.55,
    #     wg_width=0.5,
    #     wg_thickness=0.22,
    #     slab_thickness=0.0,
    #     ncore=si,
    #     nclad=sio2,
    # )
    # plt.plot(r, integral / max(integral), ".")
    # plt.xlabel("bend radius (um)")
    # plt.show()

    # rib = find_modes(
    #     wavelength=1.55, wg_width=0.5, wg_thickness=0.22, slab_thickness=0.15, ncore=3.4, nclad=1.44
    # )
    # nitride = find_modes(
    #     wavelength=1.55,
    #     wg_width=1.0,
    #     wg_thickness=0.4,
    #     ncore=2.0,
    #     nclad=sio2,
    # )

    # nitride.plot_index()
    # nitride.plot_Ex(index=0)
