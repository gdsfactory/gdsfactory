import pathlib

import gdsfactory as gf
from gdsfactory.simulation.gmpb.find_coupling_vs_gap import find_coupling_vs_gap
from gdsfactory.simulation.gmpb.get_mode_solver_coupler import get_mode_solver_coupler

PATH = pathlib.Path(__file__).parent.absolute() / "modes"


get_mode_solver_coupler_strip = gf.partial(
    get_mode_solver_coupler,
    wg_widths=(0.5, 0.5),
    slab_thickness=0.0,
)

get_mode_solver_coupler_rib90 = gf.partial(
    get_mode_solver_coupler,
    wg_widths=(0.5, 0.5),
    slab_thickness=0.09,
)

get_mode_solver_coupler_nitride = gf.partial(
    get_mode_solver_coupler, wg_widths=(0.5, 0.5), slab_thickness=0.09, ncore=2.0
)


find_coupling_vs_gap_strip = gf.partial(
    find_coupling_vs_gap,
    mode_solver=get_mode_solver_coupler_strip,
)
find_coupling_vs_gap_rib = gf.partial(
    find_coupling_vs_gap, mode_solver=get_mode_solver_coupler_rib90
)
find_coupling_vs_gap_nitride = gf.partial(
    find_coupling_vs_gap, mode_solver=get_mode_solver_coupler_nitride
)


if __name__ == "__main__":
    import gdsfactory.simulation.gmpb as gm

    df = find_coupling_vs_gap_strip()
    gm.plot_coupling_vs_gap(df)
