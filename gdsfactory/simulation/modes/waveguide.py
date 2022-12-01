from __future__ import annotations

import pathlib

import gdsfactory as gf
from gdsfactory.simulation.modes.find_neff_vs_width import find_neff_vs_width

PATH = pathlib.Path(__file__).parent.absolute() / "modes"

nm = 1e-3
wg_thickness_silicon = 220 * nm
wg_thickness_nitride = 400 * nm


find_neff_vs_width_strip = gf.partial(
    find_neff_vs_width,
    wg_thickness=wg_thickness_silicon,
    slab_thickness=0,
    filepath=PATH / "neff_vs_width_strip.csv",
)
find_neff_vs_width_rib90 = gf.partial(
    find_neff_vs_width,
    wg_thickness=wg_thickness_silicon,
    slab_thickness=90 * nm,
    filepath=PATH / "neff_vs_width_rib.csv",
)

find_neff_vs_width_nitride = gf.partial(
    find_neff_vs_width,
    wg_thickness=wg_thickness_nitride,
    slab_thickness=0.0,
    ncore=2.0,
    width1=200 * nm,
    width2=1500 * nm,
    filepath=PATH / "neff_vs_width_nitride.csv",
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import gdsfactory.simulation.modes as gm

    df = find_neff_vs_width_strip()
    df = find_neff_vs_width_rib90()
    df = find_neff_vs_width_nitride()
    gm.plot_neff_vs_width(df)
    plt.show()
