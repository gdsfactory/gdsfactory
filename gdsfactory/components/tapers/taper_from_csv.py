"""Adiabatic tapers from CSV files."""

from __future__ import annotations

__all__ = [
    "taper_0p5_to_3_l36",
    "taper_from_csv",
    "taper_w10_l100",
    "taper_w10_l150",
    "taper_w10_l200",
    "taper_w11_l200",
    "taper_w12_l200",
]

import pathlib
from functools import partial
from pathlib import Path

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec

from .._schematic import taper_schematic

data = pathlib.Path(__file__).parent / "csv_data"


@gf.cell_with_module_name(schematic_function=taper_schematic, tags=["tapers"])
def taper_from_csv(
    filepath: Path = data / "taper_strip_0p5_3_36.csv",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns taper from CSV file.

    Args:
        filepath: for CSV file.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
    """
    return cf.taper_from_csv(filepath=filepath, cross_section=cross_section)


taper_0p5_to_3_l36 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_3_36.csv")
taper_w10_l100 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_10_100.csv")
taper_w10_l150 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_10_150.csv")
taper_w10_l200 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_10_200.csv")
taper_w11_l200 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_11_200.csv")
taper_w12_l200 = partial(taper_from_csv, filepath=data / "taper_strip_0p5_12_200.csv")
