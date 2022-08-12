import pathlib
import tempfile
from typing import Optional, Tuple, Union

import meep as mp
import numpy as np
import pydantic
from meep import mpb

mpb.Verbosity(0)

tmp = pathlib.Path(tempfile.TemporaryDirectory().name).parent / "meep"
tmp.mkdir(exist_ok=True)

Floats = Tuple[float, ...]


@pydantic.validate_arguments
def get_mode_solver_coupler(
    wg_width: float = 0.5,
    gap: float = 0.2,
    wg_widths: Optional[Floats] = None,
    gaps: Optional[Floats] = None,
    wg_thickness: float = 0.22,
    slab_thickness: float = 0.0,
    ncore: float = 3.47,
    nclad: float = 1.44,
    nslab: Optional[float] = None,
    ymargin: float = 2.0,
    sz: float = 2.0,
    resolution: int = 32,
    nmodes: int = 4,
    sidewall_angles: Union[Tuple[float, ...], float] = None,
) -> mpb.ModeSolver:
    """Returns mode_solver simulation.

    Args:
        wg_width: wg_width (um) for the symmetric case.
        gap: for the case of only two waveguides.
        wg_widths: list or tuple of waveguide widths.
        gaps: list or tuple of waveguide gaps.
        wg_thickness: wg thickness (um).
        slab_thickness: thickness for the waveguide slab.
        ncore: core material refractive index.
        nclad: clad material refractive index.
        nslab: Optional slab material refractive index. Defaults to ncore.
        ymargin: margin in y.
        sz: simulation region thickness (um).
        resolution: resolution (pixels/um).
        nmodes: number of modes.
        sidewall_angles: waveguide sidewall angle (degrees),
            tapers from wg_width at top of slab, upwards, to top of waveguide
            a sidewall_angle = 10, will have 80 degrees with respect to the substrate.

    ::

          _____________________________________________________
          |
          |
          |         widths[0]                 widths[1]
          |     <---------->     gaps[0]    <---------->
          |      ___________ <-------------> ___________      _
          |     |           |               |           |     |
        sz|_____|  ncore    |_______________|           |_____|
          |                                                   | wg_thickness
          |slab_thickness        nslab                        |
          |___________________________________________________|
          |
          |<--->                                         <--->
          |ymargin               nclad                   ymargin
          |____________________________________________________
          <--------------------------------------------------->
                                   sy

    """
    wg_widths = wg_widths or (wg_width, wg_width)
    gaps = gaps or (gap,)
    material_core = mp.Medium(index=ncore)
    material_clad = mp.Medium(index=nclad)
    material_slab = mp.Medium(index=nslab or ncore)

    # Define the computational cell.  We'll make x the propagation direction.
    # the other cell sizes should be big enough so that the boundaries are
    # far away from the mode field.

    sy = np.sum(wg_widths) + np.sum(gaps) + 2 * ymargin
    geometry_lattice = mp.Lattice(size=mp.Vector3(0, sy, sz))

    geometry = []

    y = -sy / 2 + ymargin

    gaps = list(gaps) + [0]
    for i, wg_width in enumerate(wg_widths):
        if sidewall_angles:
            geometry.append(
                mp.Prism(
                    vertices=[
                        mp.Vector3(y=y, z=slab_thickness),
                        mp.Vector3(y=y + wg_width, z=slab_thickness),
                        mp.Vector3(x=1, y=y + wg_width, z=slab_thickness),
                        mp.Vector3(x=1, y=y, z=slab_thickness),
                    ],
                    height=wg_thickness - slab_thickness,
                    center=mp.Vector3(
                        y=y + wg_width / 2,
                        z=slab_thickness + (wg_thickness - slab_thickness) / 2,
                    ),
                    # If only 1 angle is specified, use it for all waveguides
                    sidewall_angle=np.deg2rad(sidewall_angles)
                    if len(np.unique(sidewall_angles)) == 1
                    else np.deg2rad(sidewall_angles[i]),
                    material=material_core,
                )
            )
        else:
            geometry.append(
                mp.Block(
                    size=mp.Vector3(mp.inf, wg_width, wg_thickness),
                    material=material_core,
                    center=mp.Vector3(y=y + wg_width / 2, z=wg_thickness / 2),
                )
            )

        y += gaps[i] + wg_width

    # define the 2D blocks for the strip and substrate
    geometry += [
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, slab_thickness),
            material=material_slab,
            center=mp.Vector3(z=slab_thickness / 2),
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

    wg_widths_str = "_".join([str(i) for i in wg_widths])
    gaps_str = "_".join([str(i) for i in gaps])
    filename_prefix = (
        tmp / f"coupler_{wg_widths_str}_{gaps_str}_{wg_thickness}_{slab_thickness}"
    )

    mode_solver = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=nmodes,
        filename_prefix=str(filename_prefix),
        default_material=material_clad,
    )
    mode_solver.nmodes = nmodes
    mode_solver.info = dict(
        wg_widths=wg_widths,
        gaps=gaps,
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

    m = get_mode_solver_coupler(
        slab_thickness=90e-3,
        nslab=2,
        gap=0.5,
        wg_width=1,
        resolution=64,
        sidewall_angles=(10.0, 20.0),
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
