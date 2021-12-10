import pathlib

import gdsfactory as gf
from gdsfactory.simulation.modes.find_coupling_vs_gap import (
    find_coupling,
    find_coupling_vs_gap,
)
from gdsfactory.simulation.modes.get_mode_solver_coupler import get_mode_solver_coupler

PATH = pathlib.Path(__file__).parent.absolute() / "modes"

nm = 1e-3
wg_thickness = 220 * nm
nitride_thickness = 400 * nm
slab_thickness = 90 * nm
strip_width = 500 * nm
rib_width = 550 * nm


get_mode_solver_coupler_strip = gf.partial(
    get_mode_solver_coupler,
    wg_width=strip_width,
    wg_thickness=wg_thickness,
    slab_thickness=0.0,
)

get_mode_solver_coupler_rib90 = gf.partial(
    get_mode_solver_coupler,
    wg_width=rib_width,
    wg_thickness=wg_thickness,
    slab_thickness=slab_thickness,
)

get_mode_solver_coupler_nitride = gf.partial(
    get_mode_solver_coupler,
    wg_width=1.0,
    slab_thickness=0.0,
    ncore=2.0,
    wg_thickness=nitride_thickness,
    sz=4,
    ymargin=4,
)


find_coupling_strip = gf.partial(
    find_coupling,
    mode_solver=get_mode_solver_coupler_strip,
)
find_coupling_rib = gf.partial(find_coupling, mode_solver=get_mode_solver_coupler_rib90)
find_coupling_nitride = gf.partial(
    find_coupling, mode_solver=get_mode_solver_coupler_nitride
)


find_coupling_vs_gap_strip = gf.partial(
    find_coupling_vs_gap,
    mode_solver=get_mode_solver_coupler_strip,
)
find_coupling_vs_gap_rib = gf.partial(
    find_coupling_vs_gap, mode_solver=get_mode_solver_coupler_rib90
)
find_coupling_vs_gap_nitride = gf.partial(
    find_coupling_vs_gap,
    mode_solver=get_mode_solver_coupler_nitride,
    gap1=300 * nm,
    gap2=600 * nm,
)


if __name__ == "__main__":
    # import gdsfactory.simulation.modes as gm
    # df = find_coupling_vs_gap_strip()
    # gm.plot_coupling_vs_gap(df)

    # lc = find_coupling_vs_gap_strip()
    # lc = find_coupling_strip(gap=0.2)
    lc = find_coupling_nitride(gap=0.4)
    print(lc)
