from __future__ import annotations

import pathlib
import tempfile
from functools import partial
from typing import Dict, Optional, Union

import meep as mp
import pydantic
from meep import mpb

import gdsfactory as gf
from gdsfactory.simulation.gmeep.get_material import get_material
from gdsfactory.simulation.gmeep.get_meep_geometry import (
    get_meep_geometry_from_cross_section,
)
from gdsfactory.technology import LayerStack
from gdsfactory.typings import CrossSectionSpec

mpb.Verbosity(0)

tmp = pathlib.Path(tempfile.TemporaryDirectory().name).parent / "meep"
tmp.mkdir(exist_ok=True)


@pydantic.validate_arguments
def get_mode_solver_cross_section(
    cross_section: CrossSectionSpec = "strip",
    sy: float = 2.0,
    sz: float = 2.0,
    resolution: int = 32,
    nmodes: int = 4,
    wavelength: float = 1.55,
    clad_mat: str = "sio2",
    layer_stack: Optional[LayerStack] = None,
    dispersive: bool = False,
    material_name_to_meep: Optional[Dict[str, Union[str, float]]] = None,
    **kwargs,
) -> mpb.ModeSolver:
    """Returns a mode_solver simulation.

    Args:
        cross_section: CrossSection to solve
        sy: simulation region width (um)
        sz: simulation region height (um)
        resolution: resolution (pixels/um)
        nmodes: number of modes
        wavelength: wavelength at which to compute mode (um)
        clad_mat: cladding material around CrossSection
        layer_stack: contains layer to thickness, zmin and material.
            Defaults to active pdk.layer_stack.
        dispersive: use dispersive material models (requires higher resolution).
        material_name_to_meep: dispersive materials have a wavelength
            dependent index. Maps layer_stack names with meep material database names.

    """
    x = gf.get_cross_section(cross_section=cross_section, **kwargs)

    # Define the computational cell.  We'll make x the propagation direction.
    # the other cell sizes should be big enough so that the boundaries are
    # far away from the mode field.
    geometry_lattice = mp.Lattice(size=mp.Vector3(0, sy, sz))

    geometry = get_meep_geometry_from_cross_section(
        cross_section=x,
        layer_stack=layer_stack,
        material_name_to_meep=material_name_to_meep,
        wavelength=wavelength,
        dispersive=dispersive,
    )

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

    filename_prefix = tmp / f"cross_section_{x.name}"

    clad = get_material(
        name=clad_mat,
        wavelength=wavelength,
        material_name_to_meep=material_name_to_meep,
        dispersive=dispersive,
    )

    mode_solver = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=nmodes,
        default_material=clad,
        filename_prefix=str(filename_prefix),
        ensure_periodicity=False,
    )
    mode_solver.nmodes = nmodes
    mode_solver.info = dict(
        dispersive=dispersive,
        material_indices=[
            b.material.epsilon(1 / wavelength)[0, 0] ** 0.5 for b in geometry
        ],
        sy=sy,
        sz=sz,
        resolution=resolution,
        nmodes=nmodes,
    )
    return mode_solver


get_mode_solver_rib = partial(get_mode_solver_cross_section, cross_section="rib")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cross_section = "pn"
    s = gf.components.straight(cross_section=cross_section)

    s.show()

    m = get_mode_solver_cross_section(
        cross_section=cross_section,
        resolution=200,
    )

    m.init_params(p=mp.NO_PARITY, reset_fields=False)
    eps = m.get_epsilon()
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
