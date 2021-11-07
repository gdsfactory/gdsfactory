import dataclasses
from typing import Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from meep import mpb


@dataclasses.dataclass
class Mode:
    solver: mpb.ModeSolver
    mode_number: int
    wavelength: float
    neff: float
    ng: Optional[float] = None
    fraction_te: Optional[float] = None
    fraction_tm: Optional[float] = None
    E: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    eps: Optional[np.ndarray] = None

    def plot_eps(
        self,
        cmap: str = "viridis",
        origin="upper",
        logscale: bool = False,
        show: bool = True,
    ):
        """plot index profle"""
        plt.imshow(
            self.eps ** 0.5,
            cmap=cmap,
            origin=origin,
            aspect="auto",
        )
        plt.title("index profile")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_e(
        self,
        cmap: str = "viridis",
        origin="upper",
        logscale: bool = False,
        show: bool = True,
    ):
        E = self.E
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
        )
        plt.title("Waveguide mode $|E_x|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_ex(
        self,
        cmap: str = "viridis",
        origin="upper",
        logscale: bool = False,
        show: bool = True,
    ):
        ex = abs(self.E[:, :, 0, 2])
        ex = 10 * np.log10(ex) if logscale else ex
        plt.imshow(
            ex.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
        )
        plt.title("Waveguide mode $|E_x|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_ey(
        self,
        cmap: str = "viridis",
        origin="upper",
        logscale: bool = False,
        show: bool = True,
    ):
        ey = abs(self.E[:, :, 0, 1])
        ey = 10 * np.log10(ey) if logscale else ey
        plt.imshow(
            ey.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
        )
        plt.title("Waveguide mode $|E_y|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_ez(
        self,
        cmap: str = "viridis",
        origin="upper",
        logscale: bool = False,
        show: bool = True,
    ):
        E = self.E
        ez = abs(E[:, :, 0, 0])
        ez = 10 * np.log10(ez) if logscale else ez
        plt.imshow(
            ez.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
        )
        plt.title("Waveguide mode $|E_z|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_e_all(
        self,
        cmap: str = "viridis",
        origin="upper",
        logscale: bool = False,
        show: bool = True,
    ):
        plt.subplot(2, 3, 1)
        self.plot_e(show=False)

        plt.subplot(2, 3, 2)
        self.plot_ey(show=False)

        plt.subplot(2, 3, 3)
        self.plot_ex(show=False)

        plt.subplot(2, 3, 4)
        self.plot_ez(show=False)

        plt.subplot(2, 3, 5)
        self.plot_eps(show=False)

        plt.show()


@dataclasses.dataclass
class WavelengthSweep:
    wavelength: List[float]
    neff: Dict[int, List[float]]
    ng: Dict[int, List[float]]


@dataclasses.dataclass
class WidthSweep:
    width: List[float]
    neff: Dict[int, List[float]]


ModeSolverFactory = Callable[..., mpb.ModeSolver]
ModeSolverOrFactory = Union[mpb.ModeSolver, ModeSolverFactory]


if __name__ == "__main__":
    import gdsfactory.simulation.mpb as gm

    m = gm.find_modes()
    m[1].plot_e_all()
