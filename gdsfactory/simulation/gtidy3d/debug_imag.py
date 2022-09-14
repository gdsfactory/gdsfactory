"""tidy3d mode solver.

tidy3d has a powerful open source mode solver.

tidy3d can:

- compute bend modes.
- compute mode overlaps.

TODO:
- calculate dispersion

Maybe:

- combine modes package (based on modesolverpy), MPB and tidy3d APIs

"""

import pathlib
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Union

import numpy as np
from pydantic import BaseModel, Extra
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.interpolate import griddata
from tidy3d.plugins.mode.solver import compute_modes
from typing_extensions import Literal

from gdsfactory.config import CONFIG, logger
from gdsfactory.serialization import get_hash
from gdsfactory.simulation.gtidy3d.materials import si, sin, sio2
from gdsfactory.types import PathType

nm = 1e-3


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

SETTINGS_COUPLER = set(SETTINGS + ["wg_width1", "wg_width2", "gap"])
Precision = Literal["single", "double"]
FilterPol = Literal["te", "tm"]


class Waveguide(BaseModel):
    """Waveguide Model.

    Parameters:
        wavelength: (um).
        wg_width: waveguide width.
        wg_thickness: thickness waveguide (um).
        ncore: core refractive index.
        nclad: cladding refractive index.
        dn_dict: unstructured mesh array with columns field "x", "y", "dn" of local index perturbations to be interpolated.
        slab_thickness: thickness slab (um).
        t_box: thickness BOX (um).
        t_clad: thickness cladding (um).
        xmargin: margin from waveguide edge to each side (um).
        resolution: pixels/um.
        nmodes: number of modes to compute.
        bend_radius: optional bend radius (um).
        cache: filepath for caching modes. If None does not use file cache.
        precision: single or double.
        filter_pol: te, tm or None.

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
    dn_dict: Optional[Dict] = None
    slab_thickness: float
    t_box: float = 2.0
    t_clad: float = 2.0
    xmargin: float = 1.0
    resolution: int = 100
    nmodes: int = 4
    bend_radius: Optional[float] = None
    cache: Optional[PathType] = CONFIG["modes"]
    precision: Precision = "single"
    filter_pol: Optional[FilterPol] = None

    class Config:
        """Config for Waveguide."""

        extra = Extra.allow

    @property
    def t_sim(self):
        return self.t_box + self.wg_thickness + self.t_clad

    @property
    def settings(self):
        return SETTINGS

    @property
    def w_sim(self):
        return self.wg_width + 2 * self.xmargin

    @property
    def filepath(self) -> Optional[pathlib.Path]:
        if self.cache is None:
            return
        cache = pathlib.Path(self.cache)
        cache.mkdir(exist_ok=True, parents=True)
        settings = {setting: getattr(self, setting) for setting in self.settings}
        return cache / f"{get_hash(settings)}.npz"

    def get_ncore(self, wavelength: Optional[float] = None):
        wavelength = wavelength or self.wavelength
        return self.ncore(wavelength) if callable(self.ncore) else self.ncore

    def get_nclad(self, wavelength: Optional[float] = None):
        wavelength = wavelength or self.wavelength
        return self.nclad(wavelength) if callable(self.nclad) else self.nclad

    def get_n(self, Y, Z):
        """Return index matrix for a waveguide.

        Args:
            Y: 2D array.
            Z: 2D array.

        """
        w = self.wg_width
        ncore = self.get_ncore()
        nclad = self.get_nclad()
        t_box = self.t_box
        wg_thickness = self.wg_thickness
        slab_thickness = self.slab_thickness
        t_clad = self.t_clad

        inds_core = (
            (-w / 2 <= Y) & (Y <= w / 2) & (Z >= t_box) & (Z <= t_box + wg_thickness)
        )
        inds_slab = (Z >= t_box) & (Z <= t_box + slab_thickness)

        complex_solver = False
        if isinstance(ncore, complex) or isinstance(nclad, complex):
            complex_solver = True
        elif self.dn_dict is not None:
            complex_solver = True
        if complex_solver:
            mat_dtype = np.complex128 if self.precision == "double" else np.complex64
        else:
            if self.precision == "double":
                mat_dtype = np.float64

        n = np.ones_like(Y, dtype=mat_dtype) * nclad
        n[
            (-w / 2 - t_clad / 2 <= Y)
            & (Y <= w / 2 + t_clad / 2)
            & (Z >= t_box)
            & (Z <= t_box + wg_thickness + t_clad)
        ] = nclad
        n[(Z <= 1.0 + slab_thickness + t_clad)] = nclad
        n[inds_core] = ncore
        n[inds_slab] = ncore if slab_thickness else nclad

        if self.dn_dict is not None:
            print("appending perturbation")
            print(n.dtype, self.dn_dict["dn"].dtype)
            dn = griddata(
                (self.dn_dict["x"], self.dn_dict["y"]),
                self.dn_dict["dn"],
                (Y, Z),
                method="cubic",
                fill_value=0.0,
            )
            # dk = 1E-7*np.ones_like(dn)
            # dk = griddata(
            #     (self.dn_dict["x"], self.dn_dict["y"]),
            #     self.dn_dict["dk"],
            #     (Y, Z),
            #     method="cubic",
            #     fill_value=0,
            # )
            n[inds_core] += dn[inds_core]  # + 1j*dk[inds_core]
            n[inds_slab] += dn[inds_slab]  # + 1j*dk[inds_slab]

        return n

    def compute_modes(
        self,
        overwrite: bool = False,
        with_fields: bool = True,
    ) -> None:
        """Compute modes.

        Args:
            overwrite: overwrite file cache.
            with_fields: include field data.
        """
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

        nx = self.get_n(
            Xx,
            Yx,
        )
        ny = self.get_n(
            Xy,
            Yy,
        )
        nz = self.get_n(
            Xz,
            Yz,
        )
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Xx, self.Yx, self.Xy, self.Yy, self.Xz, self.Yz = Xx, Yx, Xy, Yy, Xz, Yz

        if self.cache and self.filepath and self.filepath.exists():
            data = np.load(self.filepath)

            if with_fields:
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
                    num_modes=self.nmodes,
                    bend_radius=self.bend_radius,
                    bend_axis=1,
                    angle_theta=0.0,
                    angle_phi=0.0,
                    num_pml=(0, 0),
                    target_neff=self.get_ncore(wavelength),
                    sort_by="largest_neff",
                    precision=self.precision,
                    filter_pol=self.filter_pol,
                ),
            )
        )

        self.Ex, self.Ey, self.Ez = Ex, Ey, Ez
        self.Hx, self.Hy, self.Hz = Hx, Hy, Hz
        self.neffs = neffs

        if with_fields:
            data = dict(
                Ex=self.Ex,
                Ey=self.Ey,
                Ez=self.Ez,
                Hx=self.Hx,
                Hy=self.Hy,
                Hz=self.Hz,
                neffs=self.neffs,
            )
        else:
            data = dict(neffs=self.neffs)
        if self.filepath:
            np.savez_compressed(self.filepath, **data)
            logger.info(f"write {self.filepath} mode data to file cache.")


__all__ = (
    "Waveguide",
    "si",
    "sin",
    "sio2",
)

if __name__ == "__main__":

    # Control
    dn_dict = None
    c = Waveguide(
        wavelength=1.55,
        wg_width=500 * nm,
        wg_thickness=220 * nm,
        slab_thickness=0 * nm,
        ncore=lambda x: 3.45 + 1e-1j,
        nclad=sio2,
        cache=None,
    )
    c.compute_modes()
    print(c.neffs)

    # Interpolated grid
    def cartesian(*arrays):
        mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
        dim = len(mesh)  # number of dimensions
        elements = mesh[0].size  # number of elements, any index will do
        flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
        reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
        return reshape

    x = np.linspace(np.min(-c.w_sim / 2), np.max(c.w_sim), 100)
    y = np.linspace(0, c.t_sim, 100)
    arr = cartesian(x, y)
    dn = np.zeros_like(arr[:, 0])
    print(arr)
    dn_dict = {"x": arr[:, 0], "y": arr[:, 1], "dn": dn}
    c = Waveguide(
        wavelength=1.55,
        wg_width=500 * nm,
        wg_thickness=220 * nm,
        slab_thickness=0 * nm,
        ncore=lambda x: 3.45 + 1e-1j,
        nclad=sio2,
        cache=None,
        dn_dict=dn_dict,
    )
    c.compute_modes()
    print(c.neffs)

    # With imaginary indices
    dn = 0.1 * np.ones_like(arr[:, 0]) + 0.1j * np.random.rand(len(arr[:, 0]))
    dn_dict = {"x": arr[:, 0], "y": arr[:, 1], "dn": dn}
    c = Waveguide(
        wavelength=1.55,
        wg_width=500 * nm,
        wg_thickness=220 * nm,
        slab_thickness=0 * nm,
        ncore=si,
        nclad=sio2,
        cache=None,
        dn_dict=dn_dict,
    )
    c.compute_modes()
    print(c.neffs)

    # With problematic indices
    nx = np.load("nx.npy")
    ny = np.load("ny.npy")
    nz = np.load("nz.npy")
    x = np.load("x.npy")
    y = np.load("y.npy")

    print(nx.dtype)

    ((Ex, Ey, Ez), (Hx, Hy, Hz)), neffs = (
        x.squeeze()
        for x in compute_modes(
            eps_cross=[nx**2, ny**2, nz**2],
            coords=[x, y],
            freq=SPEED_OF_LIGHT / (1.55 * 1e-6),
            mode_spec=SimpleNamespace(
                num_modes=4,
                bend_radius=None,
                bend_axis=1,
                angle_theta=0.0,
                angle_phi=0.0,
                num_pml=(0, 0),
                target_neff=3.4750055639834683,
                sort_by="largest_neff",
                precision="double",
                filter_pol=None,
            ),
        )
    )
    print(neffs)
