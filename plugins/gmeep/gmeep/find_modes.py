"""FIXME! add sidewall angles

Compute modes of a rectangular Si strip waveguide on top of oxide.
Note that you should only pay attention, here, to the guided modes,
which are the modes whose frequency falls under the light line --
that is, frequency < beta / 1.45, where 1.45 is the SiO2 index.

Since there's no special lengthscale here, I'll just
use microns.  In general, if you use units of x, the frequencies
output are equivalent to x/lambda# so here, the freqeuncies will be
output as um/lambda, e.g. 1.5um would correspond to the frequency
1/1.5 = 0.6667.

"""
import pathlib
import tempfile
from typing import Dict, Union
import pydantic

import meep as mp
from meep import mpb

from gmeep.config import disable_print, enable_print
from gmeep.types import Mode

mpb.Verbosity(0)

tmp = pathlib.Path(tempfile.TemporaryDirectory().name).parent / "meep"
tmp.mkdir(exist_ok=True)


@pydantic.validate_arguments
def mode_solver_rib(
    wg_width: float = 0.45,
    wg_thickness: float = 0.22,
    slab_thickness: int = 0.0,
    ncore: float = 3.47,
    nclad: float = 1.44,
    sx: float = 2.0,
    sy: float = 2.0,
    res: int = 32,
    nmodes: int = 4,
):
    """Returns a mode_solver simulation.

    Args:
        wg_width: wg_width (um)
        wg_thickness: wg height (um)
        slab_thickness: thickness for the waveguide slab
        ncore: core material refractive index
        nclad: clad material refractive index
        sx: simulation region width (um)
        sy: simulation region height (um)
        res: resolution (pixels/um)
        nmodes: number of modes
    """
    material_core = mp.Medium(index=ncore)
    material_clad = mp.Medium(index=nclad)

    # Define the computational cell.  We'll make x the propagation direction.
    # the other cell sizes should be big enough so that the boundaries are
    # far away from the mode field.
    geometry_lattice = mp.Lattice(size=mp.Vector3(0, sx, sy))

    # define the 2d blocks for the strip and substrate
    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, mp.inf),
            material=material_clad,
        ),
        # uncomment this for air cladded waveguides
        # mp.Block(
        #     size=mp.Vector3(mp.inf, mp.inf, 0.5 * (sy - wg_thickness)),
        #     center=mp.Vector3(z=0.25 * (sy + wg_thickness)),
        #     material=material_clad,
        # ),
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, slab_thickness),
            material=material_core,
            center=mp.Vector3(z=-0.5 * slab_thickness),
        ),
        mp.Block(
            size=mp.Vector3(mp.inf, wg_width, wg_thickness),
            material=material_core,
            center=mp.Vector3(z=-0.5 * wg_thickness),
        ),
    ]

    # The k (i.e. beta, i.e. propagation constant) points to look at, in
    # units of 2*pi/um.  We'll look at num_k points from k_min to k_max.
    num_k = 9
    k_min = 0.1
    k_max = 3.0
    k_points = mp.interpolate(num_k, [mp.Vector3(k_min), mp.Vector3(k_max)])

    # Increase this to see more modes.  (The guided ones are the ones below the
    # light line, i.e. those with frequencies < kmag / 1.45, where kmag
    # is the corresponding column in the output if you grep for "freqs:".)
    # use this prefix for output files

    filename_prefix = tmp / f"rib_{wg_width}_{wg_thickness}_{slab_thickness}"

    mode_solver = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=res,
        num_bands=nmodes,
        filename_prefix=str(filename_prefix),
    )
    return mode_solver


@pydantic.validate_arguments
def find_modes(
    mode_solver_function=mode_solver_rib,
    wg_width: float = 0.45,
    wg_thickness: float = 0.22,
    slab_thickness: float = 0.0,
    ncore: float = 3.45,
    nclad: float = 1.45,
    sx: float = 2.0,
    sy: float = 2.0,
    res: int = 32,
    tol: float = 1e-6,
    wavelength: float = 1.55,
    mode_number: int = 1,
    parity=mp.NO_PARITY,
) -> Dict[str, Union[mpb.ModeSolver, float]]:
    """Returns effective index and group index for a mode.

    Args:
        mode_solver_function: function that returns mpb.ModeSolver
        wg_width: wg_width (um)
        wg_thickness: wg height (um)
        ncore: core material refractive index
        nclad: clad material refractive index
        sx: supercell width (um)
        sy: supercell height (um)
        res: (pixels/um)
        wavelength: wavelength
        mode_number: mode order
        paririty: mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM. Reduces spurious modes.

    Returns: Dict
        mode_solver
        neff
        ng


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
    mode_solver = mode_solver_function(
        wg_width=wg_width,
        wg_thickness=wg_thickness,
        slab_thickness=slab_thickness,
        ncore=ncore,
        nclad=nclad,
        sx=sx,
        sy=sy,
        res=res,
    )

    omega = 1 / wavelength

    # Output the x component of the Poynting vector for mode_number bands at omega
    disable_print()
    k = mode_solver.find_k(
        parity,
        omega,
        mode_number,
        mode_number,
        mp.Vector3(1),
        tol,
        omega * 2.02,
        omega * 0.01,
        omega * 10,
        mpb.output_poynting_x,
        mpb.display_yparities,
        mpb.display_group_velocities,
    )
    enable_print()
    vg = mode_solver.compute_group_velocities()
    k = k[0]
    vg = vg[0][0]
    neff = wavelength * k
    ng = 1 / vg
    return Mode(neff=neff, ng=ng, solver=mode_solver)


if __name__ == "__main__":
    r = find_modes(wg_width=0.45, wg_thickness=0.22)
    print(r)
