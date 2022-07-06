"""tidy3d mode solver.

"""
from types import SimpleNamespace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT
from tidy3d.plugins.mode.solver import compute_modes

from gdsfactory.config import logger
from gdsfactory.simulation.gtidy3d.modes import Waveguide, create_mesh, plot

nm = 1e-3


def get_n(
    Y,
    Z,
    ncore: float,
    nclad: float,
    wg_width1: float,
    wg_width2: float,
    gap: float,
    t_box: float,
    wg_thickness: float,
    slab_thickness: float,
    t_clad: float,
):
    """Return index matrix for a waveguide coupler.

    Args:
        Y: 2D array.
        Z: 2D array.
        ncore: core index.
        nclad: cladding index.
        wg_width1: in um.
        wg_width2: in um.
        gap: in um.
        t_box: box thickness in um.
        wg_thickness: in um.
        slab_thickness: in um.
        t_clad: thickness cladding in um.


    .. code::

    -w1-gap/2

        wg_width1     wg_width2
        <------->     <------->
         _______   |   _______
        |       |  |  |       |
        |       |  |  |       |
        |       |  |  |       |
        |       |  |  |       |
        |_______|  |  |_______|
                <----->
                  gap

    """
    w1 = wg_width1
    w2 = wg_width2
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
    n[(Z >= t_box) & (Z <= t_box + slab_thickness)] = ncore if slab_thickness else nclad
    return n


class WaveguideCoupler(Waveguide):
    """Waveguide coupler Model.

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
        cache: filepath for caching modes. If None does not use file cache.

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

    wg_width: Optional[float] = None
    wg_width1: float
    wg_width2: float
    gap: float

    @property
    def w_sim(self):
        return self.wg_width1 + self.wg_width2 + self.gap + 2 * self.xmargin

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
            wg_width1=self.wg_width1,
            wg_width2=self.wg_width2,
            gap=self.gap,
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
            wg_width1=self.wg_width1,
            wg_width2=self.wg_width2,
            gap=self.gap,
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
            wg_width1=self.wg_width1,
            wg_width2=self.wg_width2,
            gap=self.gap,
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
            wg_width1=self.wg_width1,
            wg_width2=self.wg_width2,
            gap=self.gap,
            ncore=self.get_ncore(wavelength),
            nclad=self.get_nclad(wavelength),
            t_box=self.t_box,
            slab_thickness=self.slab_thickness,
            wg_thickness=self.wg_thickness,
            t_clad=self.t_clad,
        )
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Xx, self.Yx, self.Xy, self.Yy, self.Xz, self.Yz = Xx, Yx, Xy, Yy, Xz, Yz

        if self.cache and self.filepath and self.filepath.exists():
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
        if self.filepath:
            np.savez_compressed(self.filepath, **data)
            logger.info(f"write {self.filepath} mode data to file cache.")


if __name__ == "__main__":
    c = WaveguideCoupler(
        wavelength=1.55,
        wg_width1=500 * nm,
        wg_width2=500 * nm,
        gap=200 * nm,
        wg_thickness=220 * nm,
        slab_thickness=0 * nm,
        ncore=3.4,
        nclad=1.4,
    )
    # c.plot_index()
    c.plot_Ex(0)
    # plt.show()
