import pathlib

import gdsfactory as gf
import gdsfactory.simulation.gmpb as gm

PATH = pathlib.Path(__file__).parent.absolute() / "modes"

wg_thickness_silicon = 0.22
wg_thickness_nitride = 0.4


mode_solver_strip = gf.partial(
    gm.get_mode_solver_rib, wg_thickness=wg_thickness_silicon, slab_thickness=0
)
mode_solver_rib90 = gf.partial(
    gm.get_mode_solver_rib,
    wg_thickness=wg_thickness_silicon,
    slab_thickness=0.01,
    wg_width=0.55,
)

mode_solver_nitride = gf.partial(
    gm.get_mode_solver_rib,
    wg_thickness=wg_thickness_nitride,
    slab_thickness=0.0,
    wg_width=1.0,
    ncore=2.0,
)

neff_vs_width_strip = gf.partial(
    gm.find_neff_vs_width,
    mode_solver=mode_solver_strip,
    filepath=PATH / "neff_vs_width_strip.csv",
)
neff_vs_width_rib90 = gf.partial(
    gm.find_neff_vs_width,
    mode_solver=mode_solver_rib90,
    filepath=PATH / "neff_vs_width_rib.csv",
)

neff_vs_width_nitride = gf.partial(
    gm.find_neff_vs_width,
    mode_solver=mode_solver_nitride,
    w1=0.2,
    w2=1.5,
    filepath=PATH / "neff_vs_width_nitride.csv",
)


if __name__ == "__main__":
    df = neff_vs_width_strip()
    df = neff_vs_width_rib90()
    df = neff_vs_width_nitride()
