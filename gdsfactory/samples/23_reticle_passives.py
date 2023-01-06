"""Write a sample reticle together with GDS file."""

from __future__ import annotations

import pandas as pd

import gdsfactory as gf
from gdsfactory.types import Component


def mzi_te(**kwargs) -> Component:
    gc = gf.c.grating_coupler_elliptical_tm()
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_single(c, grating_coupler=gc)
    return c


def test_mask() -> Component:
    c = gf.grid(
        [
            mzi_te(),
            mzi_te(),
            gf.functions.rotate(mzi_te),
        ]
    )
    gdspath = c.write_gds("mask.gds")
    csvpath = gf.labels.write_labels.write_labels_gdstk(gdspath)
    labels = pd.read_csv(csvpath)
    assert len(labels) == 11, len(labels)
    return c


if __name__ == "__main__":
    c = test_mask()
    c.show()
