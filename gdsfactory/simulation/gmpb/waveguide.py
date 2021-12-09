import pathlib

import gdsfactory as gf
from gdsfactory.simulation.gmpb.find_neff_vs_width import find_neff_vs_width
from gdsfactory.simulation.gmpb.get_mode_solver_rib import get_mode_solver_rib

PATH = pathlib.Path(__file__).parent.absolute() / "modes"

wg_thickness_silicon = 0.22
wg_thickness_nitride = 0.4


get_mode_solver_strip = gf.partial(
    get_mode_solver_rib, wg_thickness=wg_thickness_silicon, slab_thickness=0
)
get_mode_solver_rib90 = gf.partial(
    get_mode_solver_rib,
    wg_thickness=wg_thickness_silicon,
    slab_thickness=0.01,
    wg_width=0.55,
)

get_mode_solver_nitride = gf.partial(
    get_mode_solver_rib,
    wg_thickness=wg_thickness_nitride,
    slab_thickness=0.0,
    wg_width=1.0,
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
    w1=0.2,
    w2=1.5,
    filepath=PATH / "neff_vs_width_nitride.csv",
)


if __name__ == "__main__":
    import gdsfactory.simulation.gmpb as gm

    df = find_neff_vs_width_strip()
    df = find_neff_vs_width_rib90()
    df = find_neff_vs_width_nitride()
    gm.plot_neff_vs_width(df)
