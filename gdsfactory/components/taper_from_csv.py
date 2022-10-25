"""Adiabatic tapers from CSV files."""
import pathlib
from functools import partial
from pathlib import Path

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.types import CrossSectionSpec

data = pathlib.Path(__file__).parent / "csv_data"


@gf.cell
def taper_from_csv(
    filepath: Path = data / "taper_strip_0p5_3_36.csv",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns taper from CSV file.

    Args:
        filepath: for CSV file.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
    """
    import pandas as pd

    taper_data = pd.read_csv(filepath)
    xs = taper_data["x"].values * 1e6
    ys = np.round(taper_data["width"].values * 1e6 / 2.0, 3)

    x = gf.get_cross_section(cross_section)
    layer = x.layer

    c = gf.Component()
    c.add_polygon(list(zip(xs, ys)) + list(zip(xs, -ys))[::-1], layer=layer)

    # for cladding_layer, cladding_offset in zip(x.cladding_layers, x.cladding_offsets):
    #     ys_trench = ys + cladding_offset
    #     c.add_polygon(
    #         list(zip(xs, ys_trench)) + list(zip(xs, -ys_trench))[::-1],
    #         layer=cladding_layer,
    #     )

    c.add_port(
        name="o1",
        center=(xs[0], 0),
        width=2 * ys[0],
        orientation=180,
        layer=layer,
        cross_section=x,
    )
    c.add_port(
        name="o2",
        center=(xs[-1], 0),
        width=2 * ys[-1],
        orientation=0,
        layer=layer,
        cross_section=x,
    )
    if x.add_bbox:
        c = x.add_bbox(c)
    if x.add_pins:
        c = x.add_pins(c)
    return c


taper_0p5_to_3_l36 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_3_36.csv")
taper_w10_l100 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_10_100.csv")
taper_w10_l150 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_10_150.csv")
taper_w10_l200 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_10_200.csv")
taper_w11_l200 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_11_200.csv")
taper_w12_l200 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_12_200.csv")


if __name__ == "__main__":
    # c = taper_0p5_to_3_l36()
    c = taper_w10_l100()
    # c = taper_w11_l200()
    c.show(show_ports=True)
