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

import itertools as it
import pathlib
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from pydantic import BaseModel, Extra
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.interpolate import griddata
from tidy3d.plugins.mode.solver import compute_modes
from tqdm.auto import tqdm
from typing_extensions import Literal

from gdsfactory.config import CONFIG, logger
from gdsfactory.serialization import get_hash
from gdsfactory.simulation.gtidy3d.materials import si, sin, sio2
from gdsfactory.types import PathType

nm = 1e-3


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
    """Plot waveguide index or mode in matplotlib.

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
        mat_dtype = np.float32
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
            dn = griddata(
                (self.dn_dict["x"], self.dn_dict["y"]),
                self.dn_dict["dn"],
                (Y, Z),
                method="cubic",
                fill_value=0.0,
            )
            n[inds_core] += dn[inds_core]
            n[inds_slab] += dn[inds_slab]

        return n

    def plot_index(self) -> None:
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
        plot(Xx, Yx, nx)
        plt.show()

    def compute_modes(
        self,
        overwrite: bool = False,
        with_fields: bool = True,
        isolate: bool = False,
    ) -> None:
        """Compute modes.

        Args:
            overwrite: overwrite file cache.
            with_fields: include field data.
            isolate: whether to run the solver in this interpreter (False) or a separate one (True)
            temp_dir: if isolate, which directory to save temporary files to
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

        if isolate:
            # TODO: make a package that does this automatically
            import pickle

            # Setup paths
            temp_dir = Path.cwd() / "temp"
            temp_dir.mkdir(exist_ok=True, parents=True)
            args_file_str = "args"
            argsfile = temp_dir / args_file_str
            argsfile = argsfile.with_suffix(".pkl")
            script_file_str = "script"
            scriptfile = temp_dir / script_file_str
            scriptfile = scriptfile.with_suffix(".py")
            outputs_file_str = "outputs"
            outputsfile = temp_dir / outputs_file_str
            outputsfile = outputsfile.with_suffix(".pkl")
            arguments_dict = {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "x": x,
                "y": y,
                "SPEED_OF_LIGHT": SPEED_OF_LIGHT,
                "wavelength": wavelength,
                "nmodes": self.nmodes,
                "bend_radius": self.bend_radius,
                "bend_axis": 1,
                "angle_theta": 0.0,
                "angle_phi": 0.0,
                "num_pml": (0, 0),
                "target_neff": self.get_ncore(wavelength),
                "sort_by": "largest_neff",
                "precision": self.precision,
                "filter_pol": self.filter_pol,
            }
            with open(argsfile, "wb") as outp:
                pickle.dump(arguments_dict, outp, pickle.HIGHEST_PROTOCOL)
            # Write execution file
            script_lines = [
                "import pickle\n",
                "import numpy as np\n",
                "from types import SimpleNamespace\n",
                "from tidy3d.plugins.mode.solver import compute_modes\n\n",
                'if __name__ == "__main__":\n\n',
                f"\twith open(\"{argsfile}\", 'rb') as inp:\n",
                "\t\targuments_dict = pickle.load(inp)\n\n",
            ]
            script_lines.extend(
                f'\t{key} = arguments_dict["{key}"]\n' for key in arguments_dict
            )
            script_lines.extend(
                [
                    "\t((Ex, Ey, Ez), (Hx, Hy, Hz)), neffs = (\n",
                    "\t    x.squeeze()\n",
                    "\t    for x in compute_modes(\n",
                    "\t        eps_cross=[nx**2, ny**2, nz**2],\n",
                    "\t        coords=[x, y],\n",
                    "\t        freq=SPEED_OF_LIGHT / (wavelength * 1e-6),\n",
                    "\t        mode_spec=SimpleNamespace(\n",
                    "\t            num_modes=nmodes,\n",
                    "\t            bend_radius=bend_radius,\n",
                    "\t            bend_axis=bend_axis,\n",
                    "\t            angle_theta=angle_theta,\n",
                    "\t            angle_phi=angle_phi,\n",
                    "\t            num_pml=num_pml,\n",
                    "\t            target_neff=target_neff,\n",
                    "\t            sort_by=sort_by,\n",
                    "\t            precision=precision,\n",
                    "\t            filter_pol=filter_pol,\n",
                    "\t        ),\n",
                    "\t    )\n",
                    "\t)\n",
                ]
            )
            script_lines.extend(
                [
                    f'\toutputsfile = "{outputsfile}"\n',
                    "\toutputs_dict = {\n",
                    '\t\t    "Ex": Ex,\n',
                    '\t\t    "Ey": Ey,\n',
                    '\t\t    "Ez": Ez,\n',
                    '\t\t    "Hx": Hx,\n',
                    '\t\t    "Hy": Hy,\n',
                    '\t\t    "Hz": Hz,\n',
                    '\t\t    "neffs": neffs,\n',
                    "\t\t}\n",
                    '\twith open(outputsfile, "wb") as outp:\n',
                    "\t\t    pickle.dump(outputs_dict, outp, pickle.HIGHEST_PROTOCOL)\n",
                ]
            )
            with open(scriptfile, "w") as script_file_obj:
                script_file_obj.writelines(script_lines)
            with subprocess.Popen(
                ["python", scriptfile],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as proc:
                if not proc.stderr:
                    while not outputsfile.exists():
                        print(proc.stdout.read().decode())
                        print(proc.stderr.read().decode())
                        sys.stdout.flush()
                        sys.stderr.flush()
                        time.sleep(1)
                logger.info(f"python {scriptfile}")

            with open(outputsfile, "rb") as inp:
                outputs_dict = pickle.load(inp)

            Ex = outputs_dict["Ex"]
            Ey = outputs_dict["Ey"]
            Ez = outputs_dict["Ez"]
            Hx = outputs_dict["Hx"]
            Hy = outputs_dict["Hy"]
            Hz = outputs_dict["Hz"]
            neffs = outputs_dict["neffs"]

            import shutil

            shutil.rmtree(temp_dir)

        else:  # legacy
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

    def compute_mode_properties(self) -> Tuple[List[float], List[float], List[float]]:
        """Computes mode areas, fraction_te and fraction_tm."""
        if not hasattr(self, "neffs"):
            self.compute_modes()
        mode_areas = []
        fraction_te = []
        fraction_tm = []

        for mode_index in range(self.nmodes):
            e_fields = (
                self.Ex[..., mode_index],
                self.Ey[..., mode_index],
                self.Ez[..., mode_index],
            )
            h_fields = (
                self.Hx[..., mode_index],
                self.Hy[..., mode_index],
                self.Hz[..., mode_index],
            )

            areas_e = [np.sum(np.abs(e) ** 2) for e in e_fields]
            areas_e /= np.sum(areas_e)
            areas_e *= 100

            areas_h = [np.sum(np.abs(h) ** 2) for h in h_fields]
            areas_h /= np.sum(areas_h)
            areas_h *= 100

            fraction_te.append(areas_e[0] / (areas_e[0] + areas_e[1]))
            fraction_tm.append(areas_e[1] / (areas_e[0] + areas_e[1]))

            areas = areas_e.tolist()
            areas.extend(areas_h)
            mode_areas.append(areas)

        self.mode_areas = mode_areas
        self.fraction_te = fraction_te
        self.fraction_tm = fraction_tm
        return mode_areas, fraction_te, fraction_tm

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
        """Show waveguide name."""
        return ", \n".join([f"{k} = {getattr(self, k)!r}" for k in self.settings])

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


class WaveguideCoupler(Waveguide):
    """Waveguide coupler Model.

    Parameters:
        wavelength: (um).
        wg_width1: left waveguide width in um.
        wg_width2: right waveguide width in um.
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
        cache: filepath for caching modes. If None does not use file cache.

    ::

        -w1-gap/2

            wg_width1     wg_width2
            <------->     <------->
             _______   |   _______   __
            |       |  |  |       | |
            |       |  |  |       | |
            |       |_____|       | | wg_thickness
            |slab_thickness       | |
            |_____________________| |__
                    <----->
                      gap
            <--------------------->
                     w_sim

    """

    wg_width: Optional[float] = None
    wg_width1: float
    wg_width2: float
    gap: float

    @property
    def w_sim(self):
        return self.wg_width1 + self.wg_width2 + self.gap + 2 * self.xmargin

    def get_n(self, Y, Z):
        """Return index matrix for a waveguide coupler.

        Args:
            Y: 2D array.
            Z: 2D array.

        """
        w1 = self.wg_width1
        w2 = self.wg_width2
        gap = self.gap
        ncore = self.get_ncore()
        nclad = self.get_nclad()
        t_box = self.t_box
        wg_thickness = self.wg_thickness
        slab_thickness = self.slab_thickness
        t_clad = self.t_clad

        n = np.ones_like(Y) * nclad
        n[(Z <= 1.0 + slab_thickness + t_clad)] = nclad
        n[
            (-w1 - gap / 2 <= Y)
            & (Y <= -gap / 2)
            & (Z >= t_box)
            & (Z <= t_box + wg_thickness)
        ] = ncore
        n[
            (gap / 2 <= Y)
            & (Y <= gap / 2 + w2)
            & (Z >= t_box)
            & (Z <= t_box + wg_thickness)
        ] = ncore
        n[(Z >= t_box) & (Z <= t_box + slab_thickness)] = (
            ncore if slab_thickness else nclad
        )
        return n

    @property
    def settings(self):
        return SETTINGS_COUPLER

    def find_coupling(self, power_ratio: float = 1.0) -> float:
        """Returns the coupling length (um) of the directional coupler to achieve power_ratio, where 1 means 100% power transfer."""
        if not hasattr(self, "neffs"):
            self.compute_modes()
        neff1 = self.neffs[0]
        neff2 = self.neffs[1]
        dneff = (neff1 - neff2).real
        return self.wavelength / (np.pi * dneff) * np.arcsin(np.sqrt(power_ratio))


def sweep_bend_loss(
    bend_radius_min: float = 2.0,
    bend_radius_max: float = 5,
    steps: int = 4,
    mode_index: int = 0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns overlap integral squared for the bend mode mismatch loss.

    The loss is squared because you hit the bend loss twice
    (from bend to straight and from straight to bend).

    Args:
        bend_radius_min: min bend radius (um).
        bend_radius_max: max bend radius (um).
        steps: number of steps.
        mode_index: where 0 is the fundamental mode.

    Keyword Args:
        wavelength: (um).
        wg_width: waveguide width in um.
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


def sweep_width(
    width1: float = 200 * nm,
    width2: float = 1000 * nm,
    steps: int = 12,
    nmodes: int = 4,
    **kwargs,
) -> pd.DataFrame:
    """Sweep waveguide width and compute effective index.

    Args:
        width1: starting waveguide width in um.
        width2: end waveguide width in um.
        steps: number of points.
        nmodes: number of modes to compute.


    Keyword Args:
        wavelength: (um).
        wg_width: waveguide width in um.
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

    """
    width = np.linspace(width1, width2, steps)
    neff = {mode_number: [] for mode_number in range(1, nmodes + 1)}
    for wg_width in tqdm(width):
        wg = Waveguide(nmodes=nmodes, wg_width=wg_width, **kwargs)
        wg.compute_modes()
        for mode_number in range(1, nmodes + 1):
            neff[mode_number].append(np.real(wg.neffs[mode_number - 1]))

    df = pd.DataFrame(neff)
    df["width"] = width
    return df


def sweep_neff(
    wavelength: float = 1.55,
    thicknesses: Tuple[float, ...] = (220 * nm,),
    widths: Tuple[float, ...] = (500 * nm,),
    mode_index: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Sweep waveguide width and compute effective index.

    Args:
        wavelength: (um).
        thicknesses: in um.
        widths: in um.
        mode_index: integer, where 0 is the fundamental mode.

    Keyword Args:
        mode_index: integer.
        ncore: core refractive index.
        nclad: cladding refractive index.
        slab_thickness: thickness slab (um).
        t_box: thickness BOX (um).
        t_clad: thickness cladding (um).
        xmargin: margin from waveguide edge to each side (um).
        resolution: pixels/um.
        nmodes: number of modes to compute.
        bend_radius: optional bend radius (um).

    """
    widths_thicknesses = list(it.product(widths, thicknesses))

    neff = np.zeros(len(widths_thicknesses))
    w = np.zeros(len(widths_thicknesses))
    t = np.zeros(len(widths_thicknesses))

    for i, (wg_width, wg_thickness) in enumerate(tqdm(widths_thicknesses)):
        wg = Waveguide(
            wg_width=wg_width,
            wg_thickness=wg_thickness,
            wavelength=wavelength,
            **kwargs,
        )
        wg.compute_modes()
        wg.compute_mode_properties()
        neff[i] = np.real(wg.neffs[mode_index])
        w[i] = wg_width
        t[i] = wg_thickness

    return pd.DataFrame(dict(neff=neff, widths=w, thickness=t))


def group_index(
    wavelength: float, wavelength_step: float = 0.01, mode_index: int = 0, **kwargs
) -> float:
    """Returns group_index.

    Args:
        wavelength: (um).
        wavelength_step: in um.
        mode_index: integer, where 0 is the fundamental mode.

    Keyword Args:
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

    """
    wc = Waveguide(wavelength=wavelength, **kwargs)
    wf = Waveguide(
        wavelength=wavelength + wavelength_step,
        **kwargs,
    )
    wb = Waveguide(
        wavelength=wavelength - wavelength_step,
        **kwargs,
    )
    wc.compute_modes()
    wb.compute_modes()
    wf.compute_modes()

    nc = np.real(wc.neffs[mode_index])
    nb = np.real(wb.neffs[mode_index])
    nf = np.real(wf.neffs[mode_index])
    return nc - wavelength * (nf - nb) / (2 * wavelength_step)


def sweep_group_index(
    wavelength: float = 1.55,
    thicknesses: Tuple[float, ...] = (220 * nm,),
    widths: Tuple[float, ...] = (500 * nm,),
    **kwargs,
) -> pd.DataFrame:
    """Sweep waveguide width and compute group index.

    Args:
        wavelength: (um).
        thicknesses: in um.
        widths: in um.

    Keyword Args:
        mode_index: integer.
        ncore: core refractive index.
        nclad: cladding refractive index.
        slab_thickness: thickness slab (um).
        t_box: thickness BOX (um).
        t_clad: thickness cladding (um).
        xmargin: margin from waveguide edge to each side (um).
        resolution: pixels/um.
        nmodes: number of modes to compute.
        bend_radius: optional bend radius (um).

    """
    widths_thicknesses = list(it.product(widths, thicknesses))

    ng = np.zeros(len(widths_thicknesses))
    w = np.zeros(len(widths_thicknesses))
    t = np.zeros(len(widths_thicknesses))

    for i, (wg_width, wg_thickness) in enumerate(tqdm(widths_thicknesses)):
        ng[i] = group_index(
            wavelength=wavelength,
            wg_width=wg_width,
            wg_thickness=wg_thickness,
            **kwargs,
        )
        w[i] = wg_width
        t[i] = wg_thickness

    return pd.DataFrame(dict(ng=ng, widths=w, thickness=t))


def plot_sweep_width(
    width1: float = 200 * nm,
    width2: float = 1000 * nm,
    steps: int = 12,
    nmodes: int = 4,
    cmap: str = "magma",
    **kwargs,
) -> None:
    """Sweep waveguide width and compute effective index.

    Args:
        width1: starting waveguide width in um.
        width2: end waveguide width in um.
        steps: number of points.
        nmodes: number of modes to compute.
        cmap: colormap for the TE fraction.

    Keyword Args:
        wavelength: (um).
        wg_width: waveguide width in um.
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

    """
    width = np.linspace(width1, width2, steps)
    neff = {mode_number: [] for mode_number in range(nmodes)}

    for wg_width in tqdm(width):
        wg = Waveguide(nmodes=nmodes, wg_width=wg_width, **kwargs)
        wg.compute_modes()
        wg.compute_mode_properties()
        for mode_number in range(nmodes):
            n = np.real(wg.neffs[mode_number])
            neff[mode_number].append(n)

            fraction_te = wg.fraction_te[mode_number]
            plt.scatter(wg_width, n, c=fraction_te, vmin=0, vmax=1, cmap=cmap)

    for mode_number in range(nmodes):
        plt.plot(width, neff[mode_number], c="gray")

    plt.colorbar().set_label("TE-Fraction")
    plt.xlabel("width (um)")
    plt.ylabel("neff")


__all__ = (
    "Waveguide",
    "plot_sweep_width",
    "si",
    "sin",
    "sio2",
    "sweep_bend_loss",
    "sweep_width",
    "sweep_neff",
    "sweep_group_index",
    "group_index",
)

if __name__ == "__main__":
    widths = np.arange(400, 601, 50) * 1e-3
    widths = np.array([500]) * nm
    thicknesses = np.array([210, 220, 230]) * nm
    widths = np.array([490, 500, 510]) * nm
    widths = np.array([495, 500, 505]) * nm
    thicknesses = np.array([220]) * nm

    df = sweep_neff(
        ncore=si,
        nclad=sio2,
        slab_thickness=0 * nm,
        thicknesses=thicknesses,
        widths=widths,
    )

    neffs = df.neff.values

    # df = sweep_group_index(
    #     ncore=si,
    #     nclad=sio2,
    #     slab_thickness=0 * nm,
    #     thicknesses=np.array([220]) * nm,
    #     widths=widths,
    # )
    # c = Waveguide(
    #     wavelength=1.55,
    #     wg_width=500 * nm,
    #     wg_thickness=220 * nm,
    #     slab_thickness=0 * nm,
    #     ncore=si,
    #     nclad=sio2,
    # )
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
    # c = WaveguideCoupler(
    #     wavelength=1.55,
    #     wg_width1=500 * nm,
    #     wg_width2=500 * nm,
    #     gap=200 * nm,
    #     wg_thickness=220 * nm,
    #     slab_thickness=100 * nm,
    #     ncore=si,
    #     nclad=sio2,
    # )
    # print(c.find_coupling())
    # c.plot_index()

    # mode_areas, te, tm = c.compute_mode_properties()
    # c.plot_Ex(0)
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

    # plot_sweep_width(
    #     steps=3,
    #     wavelength=1.55,
    #     wg_thickness=220 * nm,
    #     slab_thickness=0 * nm,
    #     ncore=si,
    #     nclad=sio2,
    # )
    # plt.show()
    # ng = group_index(
    #     wg_width=500 * nm,
    #     wavelength=1.55,
    #     wg_thickness=220 * nm,
    #     slab_thickness=0 * nm,
    #     ncore=si,
    #     nclad=sio2,
    # )
    # print(ng)
    # plt.show()
