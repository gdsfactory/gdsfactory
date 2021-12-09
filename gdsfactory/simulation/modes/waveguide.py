import pathlib

import gdsfactory as gf
from gdsfactory.simulation.modes.find_neff_vs_width import find_neff_vs_width
from gdsfactory.simulation.modes.get_mode_solver_rib import get_mode_solver_rib

PATH = pathlib.Path(__file__).parent.absolute() / "modes"

nm = 1e-3
wg_thickness_silicon = 220 * nm
wg_thickness_nitride = 400 * nm


get_mode_solver_strip = gf.partial(
    get_mode_solver_rib, wg_thickness=wg_thickness_silicon, slab_thickness=0
)
get_mode_solver_rib90 = gf.partial(
    get_mode_solver_rib,
    wg_thickness=wg_thickness_silicon,
    slab_thickness=90 * nm,
    wg_width=550 * nm,
)

get_mode_solver_nitride = gf.partial(
    get_mode_solver_rib,
    wg_thickness=wg_thickness_nitride,
    slab_thickness=0.0,
    wg_width=1000 * nm,
    ncore=2.0,
)

find_neff_vs_width_strip = gf.partial(
    find_neff_vs_width,
    mode_solver=get_mode_solver_strip,
    filepath=PATH / "neff_vs_width_strip.csv",
)
find_neff_vs_width_rib90 = gf.partial(
    find_neff_vs_width,
    mode_solver=get_mode_solver_rib90,
    filepath=PATH / "neff_vs_width_rib.csv",
)

find_neff_vs_width_nitride = gf.partial(
    find_neff_vs_width,
    mode_solver=get_mode_solver_nitride,
    w1=200 * nm,
    w2=1500 * nm,
    filepath=PATH / "neff_vs_width_nitride.csv",
)


if __name__ == "__main__":
    import gdsfactory.simulation.modes as gm

    df = find_neff_vs_width_strip()
    df = find_neff_vs_width_rib90()
    df = find_neff_vs_width_nitride()
    gm.plot_neff_vs_width(df)
