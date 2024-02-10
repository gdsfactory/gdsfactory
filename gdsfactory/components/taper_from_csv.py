"""Adiabatic tapers from CSV files."""
from __future__ import annotations

import pathlib
from pathlib import Path

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import Callable, CrossSectionSpec

data = pathlib.Path(__file__).parent / "csv_data"


@gf.cell
def taper_from_csv(
    filepath: Path = data / "taper_strip_0p5_3_36.csv",
    cross_section: CrossSectionSpec = "xs_sc",
    post_process: Callable | None = None,
) -> Component:
    """Returns taper from CSV file.

    Args:
        filepath: for CSV file.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        post_process: function to post process the component.
    """
    import pandas as pd

    taper_data = pd.read_csv(filepath)
    xs = taper_data["x"].values * 1e6
    ys = np.round(taper_data["width"].values * 1e6 / 2.0, 3)

    x = gf.get_cross_section(cross_section)
    layer = x.layer

    c = gf.Component()
    c.add_polygon(list(zip(xs, ys)) + list(zip(xs, -ys))[::-1], layer=layer)

    for section in x.sections[1:]:
        ys_trench = ys + section.width
        c.add_polygon(
            list(zip(xs, ys_trench)) + list(zip(xs, -ys_trench))[::-1],
            layer=section.layer,
        )

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
        x.add_bbox(c)
    if x.add_pins:
        c = x.add_pins(c)

    if post_process:
        post_process(c)
    return c


if __name__ == "__main__":
    c = taper_from_csv(cross_section=gf.cross_section.xs_rc_bbox)
    c.show(show_ports=True)
