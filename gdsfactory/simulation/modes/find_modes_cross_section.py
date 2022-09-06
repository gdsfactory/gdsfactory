"""Compute modes of a rectangular Si strip waveguide on top of oxide. Note that \
you should only pay attention, here, to the guided modes, which are the modes \
whose frequency falls under the light line -- that is, frequency < beta / 1.45, \
where 1.45 is the SiO2 index.

Since there's no special lengthscale here, you can just
use microns. In general, if you use units of x, the frequencies
output are equivalent to x/lambda# so here, the frequencies will be
output as um/lambda, e.g. 1.5um would correspond to the frequency
1/1.5 = 0.6667.

"""
import pathlib
import pickle
from typing import Dict, Optional

import meep as mp
import numpy as np
from meep import mpb

from gdsfactory.config import CONFIG
from gdsfactory.simulation.disable_print import disable_print, enable_print
from gdsfactory.simulation.get_sparameters_path import get_kwargs_hash
from gdsfactory.simulation.modes.get_mode_solver_cross_section import (
    get_mode_solver_cross_section,
)
from gdsfactory.simulation.modes.types import Mode
from gdsfactory.types import CrossSectionSpec, PathType

mpb.Verbosity(0)


def find_modes_cross_section(
    cross_section: CrossSectionSpec,
    tol: float = 1e-6,
    wavelength: float = 1.55,
    mode_number: int = 1,
    parity=mp.NO_PARITY,
    cache: Optional[PathType] = CONFIG["modes"],
    overwrite: bool = False,
    **kwargs,
) -> Dict[int, Mode]:
    """Computes mode effective and group index for a CrossSection.

    Args:
        cross_section: CrossSection to solve
        tol: tolerance when finding modes.
        wavelength: wavelength in um.
        mode_number: mode order of the first mode.
        parity: mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM.
        cache: directory path to cache modes. None disables the file cache.
        overwrite: forces simulating again.
        kwargs: waveguide settings.

    Keyword Args:
        cross_section: CrossSection to simulate
        sy: simulation region width (um).
        sz: simulation region height (um).
        resolution: resolution (pixels/um).
        nmodes: number of modes to compute.
        wavelength: wavelength at which to compute mode (um)
        clad_mat: cladding material around CrossSection
        layer_stack: contains layer to thickness, zmin and material.
            Defaults to active pdk.layer_stack.
        dispersive: use dispersive material models (requires higher resolution).
        material_name_to_meep: dispersive materials have a wavelength
            dependent index. Maps layer_stack names with meep material database names.

    Returns: Dict[mode_number, Mode]

    compute mode_number of the lowest frequencies as a function of k. Also display
    "parities", i.e. whether the mode is symmetric or anti_symmetric
    through the y=0 and z=0 planes.
    mode_solver.run(mpb.display_yparities, mpb.display_zparities)

    Above, we outputted the dispersion relation: frequency (omega) as a
    function of wavevector kx (beta). Alternatively, you can compute
    beta for a given omega -- for example, you might want to find the
    modes and wavevectors at a fixed wavelength of 1.55 microns. You
    can do that using the find_k function:

    """
    modes = {}
    mode_solver = get_mode_solver_cross_section(
        cross_section=cross_section, wavelength=wavelength, **kwargs
    )

    nmodes = mode_solver.nmodes
    omega = 1 / wavelength

    h = get_kwargs_hash(
        wavelength=wavelength,
        parity=parity,
        **kwargs,
    )

    if cache:
        cache = pathlib.Path(cache)
        cache.mkdir(exist_ok=True, parents=True)
        filepath = cache / f"{h}_{mode_number}.pkl"

        if filepath.exists() and not overwrite:
            for index, i in enumerate(range(mode_number, mode_number + nmodes)):
                filepath = cache / f"{h}_{index}.pkl"
                mode = pickle.loads(filepath.read_bytes())
                modes[i] = mode
            return modes

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
    neff = np.array(k) * wavelength

    # vg = mode_solver.compute_group_velocities()
    # vg = vg[0]
    # ng = 1 / np.array(vg)

    for index, i in enumerate(range(mode_number, mode_number + nmodes)):
        Ei = mode_solver.get_efield(i)
        Hi = mode_solver.get_hfield(i)
        y_num = np.shape(Ei[:, :, 0, 0])[0]
        z_num = np.shape(Ei[:, :, 0, 0])[1]

        modes[i] = Mode(
            mode_number=i,
            neff=neff[index],
            wavelength=wavelength,
            E=Ei,
            H=Hi,
            eps=mode_solver.get_epsilon().T,
            y=np.linspace(
                -1 * mode_solver.info["sy"] / 2.0,
                mode_solver.info["sy"] / 2.0,
                y_num,
            ),
            z=np.linspace(
                -1 * mode_solver.info["sz"] / 2.0,
                mode_solver.info["sz"] / 2.0,
                z_num,
            ),
            material_indices=mode_solver.info["material_indices"],
        )
        if cache:
            filepath = cache / f"{h}_{index}.pkl"
            filepath.write_bytes(pickle.dumps(modes[i]))

    return modes


if __name__ == "__main__":
    import gdsfactory as gf

    cross_section = "strip"
    s = gf.components.straight(cross_section=cross_section)

    s.show()

    m = find_modes_cross_section(
        cross_section=cross_section,
    )

    print(m)

    m1 = m[1]

    print(f"Mode 1 n_eff: {m1.neff}")

    m1.plot_e_all()
    # print(f'Mode 1 n_g: {m1.ng}')

    # print(np.shape(m1.y))
    # print(np.shape(m1.z))
    # print(np.shape(m1.E[:, :, 0, 1]))

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
