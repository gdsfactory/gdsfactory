"""Adiabatic tapers from CSV files."""

from __future__ import annotations

__all__ = ["taper_from_csv"]

from pathlib import Path

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import PATH
from gdsfactory.typings import CrossSectionSpec

# The CSV files live next to the gdsfactory.components cell.
data = PATH.module / "components" / "tapers" / "csv_data"


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
    xs: list[float] = taper_data["x"].values * 1e6
    ys: npt.NDArray[np.float64] = np.round(taper_data["width"].values * 1e6 / 2.0, 3)

    x = gf.get_cross_section(cross_section)
    layer = x.layer

    c = gf.Component()
    c.add_polygon(
        list(zip(xs, ys, strict=False)) + list(zip(xs, -ys, strict=False))[::-1],
        layer=layer,
    )

    for section in x.get_sections()[1:]:
        ys_trench = ys + section.width
        c.add_polygon(
            [(float(x), float(y)) for x, y in zip(xs, ys_trench, strict=False)]
            + [(float(x), float(y)) for x, y in zip(xs, -ys_trench, strict=False)][
                ::-1
            ],
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
    x.add_bbox(c)
    return c
