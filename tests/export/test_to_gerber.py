"""Tests for Gerber export.

Covers the three failures in https://github.com/gdsfactory/gdsfactory/issues/4748:
layer-key lookup, `%FS` command order, and numpy polygon vertices.
"""

from pathlib import Path

import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory.export.to_gerber import (
    GerberLayer,
    GerberOptions,
    decimal_digits,
    format_specification,
    number,
    polygon,
    to_gerber,
)

LAYER = (1, 0)
GERBER_LAYER = GerberLayer(
    name="F_Cu",
    function=["Copper", "L1", "Top"],
    polarity="Positive",
)


def _rectangle() -> gf.Component:
    component = gf.Component()
    component.add_polygon([(0, 0), (10, 0), (10, 5), (0, 5)], layer=LAYER)
    return component


def test_format_specification_matches_ucamco() -> None:
    """Issue 4748 bug 2: `%FSLA46Y46X*%` is invalid; `%FSLAX46Y46*%` is not."""
    assert format_specification(4, 6) == "%FSLAX46Y46*%\n"
    assert "Y46X" not in format_specification(4, 6)


def test_number_scale_matches_format_specification() -> None:
    """Coordinate encoding must use 10**decimal_digits, not a hardcoded 10_000."""
    assert number(10.0, decimal_digits=4) == "100000"
    assert number(10.0, decimal_digits=6) == "10000000"
    assert number(-1.5, decimal_digits=4) == "-15000"


def test_polygon_accepts_numpy_vertices() -> None:
    """Issue 4748 bug 3: get_polygons_points() returns Nx2 numpy arrays."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    text = polygon(verts, decimal_digits=4)
    assert text.startswith("G36*")
    assert "X0Y0D02*" in text
    assert "X10000Y0D01*" in text
    assert text.endswith("G37*\n\n")


def test_decimal_digits_rejects_unknown_resolution() -> None:
    with pytest.raises(ValueError, match="Unsupported Gerber resolution"):
        decimal_digits(1e-9)


def test_to_gerber_writes_spec_compliant_file(tmp_path: Path) -> None:
    """End-to-end: polygons are emitted, `%FS` is valid, file terminates."""
    to_gerber(
        _rectangle(),
        tmp_path,
        {LAYER: GERBER_LAYER},
        options=GerberOptions(resolution=1e-6, int_size=4),
    )
    path = tmp_path / "F_Cu.gbr"
    assert path.exists()
    text = path.read_text()

    assert "%FSLAX46Y46*%" in text
    assert "%FSLA46Y46X*%" not in text
    assert text.count("%LPD*%\n") == 1
    assert "%MOMM*%" in text
    assert "G36*" in text
    assert "G37*" in text
    # 10 um (user units) at 6 decimal digits → X10000000
    assert "X0Y0D02*" in text
    assert "X10000000Y0D01*" in text
    assert "X10000000Y5000000D01*" in text
    assert text.rstrip().endswith("M02*")


def test_to_gerber_uses_tuple_layer_keys(tmp_path: Path) -> None:
    """Issue 4748 bug 1: default by='index' misses tuple layermap keys."""
    component = _rectangle()
    by_index = component.get_polygons_points()
    by_tuple = component.get_polygons_points(by="tuple")
    assert LAYER not in by_index
    assert LAYER in by_tuple
    assert len(by_tuple[LAYER]) == 1

    to_gerber(component, tmp_path, {LAYER: GERBER_LAYER})
    text = (tmp_path / "F_Cu.gbr").read_text()
    assert "G36*" in text, (
        "export wrote an empty Gerber because layer keys did not match"
    )


def test_to_gerber_default_options_are_a_model(tmp_path: Path) -> None:
    """Calling without options must not pass a pydantic FieldInfo."""
    to_gerber(_rectangle(), tmp_path, {LAYER: GERBER_LAYER})
    assert (tmp_path / "F_Cu.gbr").exists()
