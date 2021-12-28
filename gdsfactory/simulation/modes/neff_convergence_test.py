"""Given a `gm.get_mode_solver_` object, repeats the mode simulation with increasing hyperparameter value (`sy`, `sz`, and `res`) until the change in output is below some tolerance. Output the results of `find_modes` and the smallest hyperparameters that allowed convergence.
"""
from typing import Dict, Tuple

import meep as mp
import numpy as np

from gdsfactory.simulation.modes.find_modes import find_modes
from gdsfactory.simulation.modes.get_mode_solver_rib import get_mode_solver_rib
from gdsfactory.simulation.modes.types import Mode, ModeSolverOrFactory


def neff_domain_convergence_test(
    mode_solver: ModeSolverOrFactory = get_mode_solver_rib,
    tol: float = 1e-6,
    wavelength: float = 1.55,
    mode_number: int = 1,
    parity=mp.NO_PARITY,
    rel_conv_tol: float = 1e-6,
    rel_conv_step: float = 2e-1,
    stdout: bool = False,
    **kwargs
) -> Tuple[Dict[int, Mode], float, float, int]:
    """Repeats a find_modes on a mode_solver, increasing hyperparameters sy, sz, and res until results are no longer affected by the choice (according to conv_tol).

    Args:
        mode_solver: function that returns mpb.ModeSolver
        tol: tolerance when finding modes
        wavelength: wavelength
        mode_number: mode order of the first mode
        parity: mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM.
        rel_conv_tol: relative tolerance on hyperparameters (# decimal places; cannot be too demanding, since solver is iterative)
        rel_conv_step: relative increase in hyperparameters from initial values
        stdout: convergence test output

    Keyword Args:
        wg_width: wg_width (um)
        wg_thickness: wg height (um)
        slab_thickness: thickness for the waveguide slab
        ncore: core material refractive index
        nclad: clad material refractive index
        sy: INITIAL simulation region width (um)
        sz: INITIAL simulation region height (um)
        res: INITIAL resolution (pixels/um)
        nmodes: number of modes
        mode_solver: function that returns mpb.ModeSolver,
        tol: tolerance when finding modes,
        wavelength: wavelength,
        mode_number: mode order of the first mode,
        parity= mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM.

    Returns: Dict[mode_number, Mode], sy, sz, res
    """
    # Initial parameters
    if "sy" in kwargs:
        sy = kwargs.get("sy")
    else:
        sy = 2.0
    if "sz" in kwargs:
        sz = kwargs.get("sz")
    else:
        sz = 2.0
    if "res" in kwargs:
        res = kwargs.get("res")
    else:
        res = 32

    # Initial solve
    mode_solver = mode_solver(**kwargs) if callable(mode_solver) else mode_solver
    init_mode = find_modes(mode_solver=mode_solver, **kwargs)[mode_number]

    # Increment the hyperparameters, and check tolerance
    # Domain size
    cur_tol = np.inf
    iter = 1
    if stdout:
        print("Domain convergence")
        print("Iteration | neff | sy | sz | tol")
        print("0 | {} | {} | {} | ---".format(init_mode.neff, sy, sz))
    while cur_tol > rel_conv_tol:
        if iter == 1:
            cur_tol = 0
        # Increment hyperparameters
        cur_sy = iter * rel_conv_step * sy + sy
        cur_sz = iter * rel_conv_step * sz + sz
        # Compute new modes
        incr_mode = find_modes(
            mode_solver=mode_solver, parity=mp.NO_PARITY, sy=cur_sy, sz=cur_sz
        )[mode_number]
        # Compute tolerance
        cur_tol = abs(init_mode.neff - incr_mode.neff) / min(
            init_mode.neff, incr_mode.neff
        )
        if stdout:
            print(
                "{} | {} | {} | {} | {}".format(
                    iter, incr_mode.neff, cur_sy, cur_sz, cur_tol
                )
            )
        iter += 1

    # Resolution, using converged cell size for initial solve
    init_mode = incr_mode
    cur_tol = np.inf
    iter = 1
    if stdout:
        print("Resolution convergence")
        print("Iteration | neff | res | tol")
        print("0 | {} | {} | ---".format(init_mode.neff, res))
    while cur_tol > rel_conv_tol:
        if iter == 1:
            cur_tol = 0
        # Increment hyperparameters
        cur_res = int(iter * rel_conv_step * res + res)
        # Compute new modes
        incr_mode = find_modes(
            mode_solver=mode_solver,
            parity=mp.NO_PARITY,
            sy=cur_sy,
            sz=cur_sz,
            res=cur_res,
        )[mode_number]
        # Compute tolerance
        cur_tol = abs(init_mode.neff - incr_mode.neff) / min(
            init_mode.neff, incr_mode.neff
        )
        if stdout:
            print("{} | {} | {} | {}".format(iter, incr_mode.neff, cur_res, cur_tol))
        iter += 1

    return (incr_mode, cur_sy, cur_sz, cur_res)


def neff_resolution_convergence_test(
    mode_solver: ModeSolverOrFactory = get_mode_solver_rib,
    tol: float = 1e-6,
    wavelength: float = 1.55,
    mode_number: int = 1,
    parity=mp.NO_PARITY,
    rel_conv_tol: float = 1e-6,
    rel_conv_step: float = 2e-1,
    stdout: bool = False,
    **kwargs
) -> Tuple[Dict[int, Mode], float, float, int]:
    """Repeats a find_modes on a mode_solver, increasing hyperparameter res until results are no longer affected by the choice (according to conv_tol).

    Args:
        mode_solver: function that returns mpb.ModeSolver
        tol: tolerance when finding modes
        wavelength: wavelength
        mode_number: mode order of the first mode
        parity: mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM.
        rel_conv_tol: relative tolerance on hyperparameters (# decimal places; cannot be too demanding, since solver is iterative)
        rel_conv_step: relative increase in hyperparameters from initial values
        stdout: convergence test output

    Keyword Args:
        wg_width: wg_width (um)
        wg_thickness: wg height (um)
        slab_thickness: thickness for the waveguide slab
        ncore: core material refractive index
        nclad: clad material refractive index
        sy: INITIAL simulation region width (um)
        sz: INITIAL simulation region height (um)
        res: INITIAL resolution (pixels/um)
        nmodes: number of modes
        mode_solver: function that returns mpb.ModeSolver,
        tol: tolerance when finding modes,
        wavelength: wavelength,
        mode_number: mode order of the first mode,
        parity= mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM.

    Returns: Dict[mode_number, Mode], sy, sz, res
    """
    # Initial parameters
    if "sy" in kwargs:
        sy = kwargs.get("sy")
    else:
        sy = 2.0
    if "sz" in kwargs:
        sz = kwargs.get("sz")
    else:
        sz = 2.0
    if "res" in kwargs:
        res = kwargs.get("res")
    else:
        res = 32

    # Initial solve
    mode_solver = mode_solver(**kwargs) if callable(mode_solver) else mode_solver
    init_mode = find_modes(mode_solver=mode_solver, **kwargs)[mode_number]

    # Increment the hyperparameters, and check tolerance
    # Domain size
    cur_tol = np.inf
    iter = 1
    if stdout:
        print("Domain convergence")
        print("Iteration | neff | sy | sz | tol")
        print("0 | {} | {} | {} | ---".format(init_mode.neff, sy, sz))
    while cur_tol > rel_conv_tol:
        if iter == 1:
            cur_tol = 0
        # Increment hyperparameters
        cur_sy = iter * rel_conv_step * sy + sy
        cur_sz = iter * rel_conv_step * sz + sz
        # Compute new modes
        incr_mode = find_modes(
            mode_solver=mode_solver, parity=mp.NO_PARITY, sy=cur_sy, sz=cur_sz
        )[mode_number]
        # Compute tolerance
        cur_tol = abs(init_mode.neff - incr_mode.neff) / min(
            init_mode.neff, incr_mode.neff
        )
        if stdout:
            print(
                "{} | {} | {} | {} | {}".format(
                    iter, incr_mode.neff, cur_sy, cur_sz, cur_tol
                )
            )
        iter += 1

    # Resolution, using converged cell size for initial solve
    init_mode = incr_mode
    cur_tol = np.inf
    iter = 1
    if stdout:
        print("Resolution convergence")
        print("Iteration | neff | res | tol")
        print("0 | {} | {} | ---".format(init_mode.neff, res))
    while cur_tol > rel_conv_tol:
        if iter == 1:
            cur_tol = 0
        # Increment hyperparameters
        cur_res = int(iter * rel_conv_step * res + res)
        # Compute new modes
        incr_mode = find_modes(
            mode_solver=mode_solver,
            parity=mp.NO_PARITY,
            sy=cur_sy,
            sz=cur_sz,
            res=cur_res,
        )[mode_number]
        # Compute tolerance
        cur_tol = abs(init_mode.neff - incr_mode.neff) / min(
            init_mode.neff, incr_mode.neff
        )
        if stdout:
            print("{} | {} | {} | {}".format(iter, incr_mode.neff, cur_res, cur_tol))
        iter += 1

    return (incr_mode, cur_sy, cur_sz, cur_res)


if __name__ == "__main__":
    mode_solver = get_mode_solver_rib(
        wg_width=0.4,
        ncore=2,
        nclad=1.44,
        wg_thickness=0.4,
        res=16,
        sy=2,
        sz=2,
        nmodes=4,
    )
    # result = neff_convergence_test(
    #     mode_solver, stdout=True, res=16, sy=2, sz=2, nmodes=4
    # )
