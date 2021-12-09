"""FIXME! add sidewall angles"""
import pathlib
import tempfile

import meep as mp
import pydantic
from meep import mpb

mpb.Verbosity(0)

tmp = pathlib.Path(tempfile.TemporaryDirectory().name).parent / "meep"
tmp.mkdir(exist_ok=True)


@pydantic.validate_arguments
def get_mode_solver_rib(
    wg_width: float = 0.45,
    wg_thickness: float = 0.22,
    slab_thickness: float = 0.0,
    ncore: float = 3.47,
    nclad: float = 1.44,
    sy: float = 2.0,
    sz: float = 2.0,
    res: int = 32,
    nmodes: int = 4,
) -> mpb.ModeSolver:
    """Returns a mode_solver simulation.

    Args:
        wg_width: wg_width (um)
        wg_thickness: wg height (um)
        slab_thickness: thickness for the waveguide slab
        ncore: core material refractive index
        nclad: clad material refractive index
        sy: simulation region width (um)
        sz: simulation region height (um)
        res: resolution (pixels/um)
        nmodes: number of modes

    ::

          __________________________
          |
          |
          |         width
          |     <---------->
          |      ___________   _ _ _
          |     |           |       |
        sz|_____|           |_______|
          |                         | wg_thickness
          |slab_thickness           |
          |_________________________|
          |
          |
          |__________________________
          <------------------------>
                        sy
    """
    material_core = mp.Medium(index=ncore)
    material_clad = mp.Medium(index=nclad)

    # Define the computational cell.  We'll make x the propagation direction.
    # the other cell sizes should be big enough so that the boundaries are
    # far away from the mode field.
    geometry_lattice = mp.Lattice(size=mp.Vector3(0, sy, sz))

    # define the 2d blocks for the strip and substrate
    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, mp.inf),
            material=material_clad,
        ),
        # uncomment this for not oxide cladded waveguides
        # mp.Block(
        #     size=mp.Vector3(mp.inf, mp.inf, 0.5 * (sz - wg_thickness)),
        #     center=mp.Vector3(z=0.25 * (sz + wg_thickness)),
        #     material=material_clad,
        # ),
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, slab_thickness),
            material=material_core,
            center=mp.Vector3(z=slab_thickness / 2),
        ),
        mp.Block(
            size=mp.Vector3(mp.inf, wg_width, wg_thickness),
            material=material_core,
            center=mp.Vector3(z=wg_thickness / 2),
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
    mode_solver.nmodes = nmodes
    mode_solver.info = dict(
        wg_width=wg_width,
        wg_thickness=wg_thickness,
        slab_thickness=slab_thickness,
        ncore=ncore,
        nclad=nclad,
        sy=sy,
        sz=sz,
        res=res,
        nmodes=nmodes,
    )
    return mode_solver


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    m = get_mode_solver_rib(slab_thickness=0.09, res=64)
    m.init_params(p=mp.NO_PARITY, reset_fields=False)
    eps = m.get_epsilon()
    cmap = "viridis"
    origin = "lower"
    plt.imshow(
        eps.T ** 0.5,
        cmap=cmap,
        origin=origin,
        aspect="auto",
    )
    plt.show()
