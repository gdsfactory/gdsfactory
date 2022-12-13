from __future__ import annotations

import pathlib
import tempfile
from typing import Optional

import meep as mp
import numpy as np
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
    nslab: Optional[float] = None,
    sy: float = 2.0,
    sz: float = 2.0,
    resolution: int = 32,
    nmodes: int = 4,
    sidewall_angle: Optional[float] = None,
) -> mpb.ModeSolver:
    """Returns a mode_solver simulation.

    Args:
        wg_width: wg_width (um).
        wg_thickness: wg thickness (um).
        slab_thickness: thickness for the waveguide slab.
        ncore: core material refractive index.
        nclad: clad material refractive index.
        nslab: Optional slab material refractive index. Defaults to ncore.
        sy: simulation region width (um).
        sz: simulation region height (um).
        resolution: resolution (pixels/um).
        nmodes: number of modes.
        sidewall_angle: waveguide sidewall angle (degrees),
            tapers from wg_width at top of slab, upwards, to top of waveguide
            with respect to the normal.
            a sidewall_angle = 10, will have 80 degrees with respect to the substrate.

    ::

          __________________________
          |
          |
          |         width
          |     <---------->
          |      ___________   _ _ _
          |     |           |       |
        sz|_____|  ncore    |_______|
          |                         | wg_thickness
          |slab_thickness    nslab  |
          |_________________________|
          |
          |        nclad
          |__________________________
          <------------------------>
                        sy

    """
    material_core = mp.Medium(index=ncore)
    material_clad = mp.Medium(index=nclad)
    material_slab = mp.Medium(index=nslab or ncore)

    # Define the computational cell.  We'll make x the propagation direction.
    # the other cell sizes should be big enough so that the boundaries are
    # far away from the mode field.
    geometry_lattice = mp.Lattice(size=mp.Vector3(0, sy, sz))

    geometry = []
    if sidewall_angle:
        geometry.append(
            mp.Prism(
                vertices=[
                    mp.Vector3(y=-wg_width / 2, z=0),
                    mp.Vector3(y=wg_width / 2, z=0),
                    mp.Vector3(x=1, y=wg_width / 2, z=0),
                    mp.Vector3(x=1, y=-wg_width / 2, z=0),
                ],
                height=wg_thickness - slab_thickness,
                center=mp.Vector3(z=0),
                sidewall_angle=np.deg2rad(sidewall_angle),
                material=material_core,
            )
        )
    else:
        geometry.append(
            mp.Block(
                size=mp.Vector3(mp.inf, wg_width, wg_thickness),
                material=material_core,
                center=mp.Vector3(z=0),
            )
        )

    geometry += [
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, slab_thickness),
            material=material_slab,
            center=mp.Vector3(z=-slab_thickness / 2),
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
        resolution=resolution,
        num_bands=nmodes,
        default_material=material_clad,
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
        resolution=resolution,
        nmodes=nmodes,
    )
    return mode_solver


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    m = get_mode_solver_rib(
        slab_thickness=0.25,
        # slab_thickness=0.,
        wg_thickness=0.5,
        resolution=64,
        nslab=2,
    )
    m.init_params(p=mp.NO_PARITY, reset_fields=False)
    eps = m.get_epsilon()
    # cmap = 'viridis'
    # cmap = "RdBu"
    cmap = "binary"
    origin = "lower"
    plt.imshow(
        eps.T**0.5,
        cmap=cmap,
        origin=origin,
        aspect="auto",
        extent=[
            -m.info["sy"] / 2,
            m.info["sy"] / 2,
            -m.info["sz"] / 2,
            m.info["sz"] / 2,
        ],
    )
    plt.colorbar()
    plt.show()
