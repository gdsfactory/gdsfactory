"""
Compute modes of a rectangular Si strip waveguide on top of oxide.
Note that you should only pay attention, here, to the guided modes,
which are the modes whose frequency falls under the light line --
that is, frequency < beta / 1.45, where 1.45 is the SiO2 index.

Since there's no special lengthscale here, you can just
use microns.  In general, if you use units of x, the frequencies
output are equivalent to x/lambda# so here, the freqeuncies will be
output as um/lambda, e.g. 1.5um would correspond to the frequency
1/1.5 = 0.6667.

"""
from typing import Dict

import meep as mp
import numpy as np
from meep import mpb

from gdsfactory.simulation.modes.disable_print import disable_print, enable_print
from gdsfactory.simulation.modes.get_mode_solver_rib import get_mode_solver_rib
from gdsfactory.simulation.modes.types import Mode, ModeSolverOrFactory

mpb.Verbosity(0)


def find_modes(
    mode_solver: ModeSolverOrFactory = get_mode_solver_rib,
    tol: float = 1e-6,
    wavelength: float = 1.55,
    mode_number: int = 1,
    parity=mp.NO_PARITY,
    **kwargs
) -> Dict[int, Mode]:
    """Computes effective index and group index for a mode.

    Args:
        mode_solver: function that returns mpb.ModeSolver
        tol: tolerance when finding modes
        wavelength: wavelength
        mode_number: mode order of the first mode
        paririty: mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM.

    Keyword Args:
        wg_width: wg_width (um)
        wg_thickness: wg height (um)
        slab_thickness: thickness for the waveguide slab
        ncore: core material refractive index
        nclad: clad material refractive index
        sy: simulation region width (um)
        sz: simulation region height (um)
        res: resolution (pixels/um)
        nmodes: number of modes

    Returns: Dict[mode_number, Mode]


    compute mode_number lowest frequencies as a function of k. Also display
    "parities", i.e. whether the mode is symmetric or anti_symmetric
    through the y=0 and z=0 planes.
    mode_solver.run(mpb.display_yparities, mpb.display_zparities)

    Above, we outputed the dispersion relation: frequency (omega) as a
    function of wavevector kx (beta).  Alternatively, you can compute
    beta for a given omega -- for example, you might want to find the
    modes and wavevectors at a fixed wavelength of 1.55 microns.  You
    can do that using the find_k function:
    """
    mode_solver = mode_solver(**kwargs) if callable(mode_solver) else mode_solver
    nmodes = mode_solver.nmodes
    omega = 1 / wavelength

    # Output the x component of the Poynting vector for mode_number bands at omega
    disable_print()
    k = mode_solver.find_k(
        parity,
        omega,
        mode_number,
        mode_number + nmodes,
        mp.Vector3(1),
        tol,
        omega * 2.02,
        omega * 0.01,
        omega * 10,
        # mpb.output_poynting_x,
        mpb.display_yparities,
        mpb.display_group_velocities,
    )
    enable_print()
    vg = mode_solver.compute_group_velocities()
    vg = vg[0]
    neff = np.array(k) * wavelength
    ng = 1 / np.array(vg)

    modes = {
        i: Mode(
            mode_number=i,
            neff=neff[index],
            wavelength=wavelength,
            ng=ng,
            E=mode_solver.get_efield(i),
            H=mode_solver.get_hfield(i),
            eps=mode_solver.get_epsilon().T,
            y=np.linspace(
                -1 * mode_solver.info["sy"] / 2,
                mode_solver.info["sy"] / 2,
                int(mode_solver.info["sy"] * mode_solver.info["res"]),
            ),
            z=np.linspace(
                -1 * mode_solver.info["sz"] / 2,
                mode_solver.info["sz"] / 2,
                int(mode_solver.info["sz"] * mode_solver.info["res"]),
            ),
        )
        for index, i in enumerate(range(mode_number, mode_number + nmodes))
    }

    return modes


if __name__ == "__main__":
    ms = get_mode_solver_rib(wg_width=0.5)
    m = find_modes(mode_solver=ms)

    # tol: float = 1e-6
    # wavelength: float = 1.55
    # mode_number: int = 1
    # parity = mp.NO_PARITY

    # mode_solver = get_mode_solver_rib()
    # nmodes = mode_solver.nmodes
    # omega = 1 / wavelength

    # # Output the x component of the Poynting vector for mode_number bands at omega
    # k = mode_solver.find_k(
    #     parity,
    #     omega,
    #     mode_number,
    #     mode_number + nmodes,
    #     mp.Vector3(1),
    #     tol,
    #     omega * 2.02,
    #     omega * 0.01,
    #     omega * 10,
    #     mpb.output_poynting_x,
    #     mpb.display_yparities,
    #     mpb.display_group_velocities,
    # )
    # enable_print()
    # vg = mode_solver.compute_group_velocities()
    # vg0 = vg[0]
    # neff = np.array(k) * wavelength
    # ng = 1 / np.array(vg0)

    # modes = {
    #     i: Mode(
    #         mode_number=i,
    #         neff=neff[index],
    #         solver=mode_solver,
    #         wavelength=wavelength,
    #     )
    #     for index, i in enumerate(range(mode_number, mode_number + nmodes))
    # }
