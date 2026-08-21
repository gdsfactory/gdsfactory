"""Tests for Gerber export.

Covers https://github.com/gdsfactory/gdsfactory/issues/4748 plus the PIC-to-PCB
handoff: um layout units, millimetre Gerber, closed G36 regions, and `.gbrjob`.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory.export.to_gerber import (
    BoardOptions,
    GerberLayer,
    GerberOptions,
    decimal_digits,
    file_unit_scale,
    format_specification,
    number,
    polygon,
    to_gerber,
)
from gdsfactory.gpdk import LAYER

WG = (1, 0)
COPPER = GerberLayer(
    name="F_Cu",
    function=["Copper", "L1", "Top"],
    polarity="Positive",
)


def _rectangle() -> gf.Component:
    component = gf.Component()
    component.add_polygon([(0, 0), (10, 0), (10, 5), (0, 5)], layer=WG)
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


def test_file_unit_scale_um_layout_to_mm_gerber() -> None:
    """100 um in GDSFactory is 0.1 mm on a Gerber board, not 100 mm."""
    assert file_unit_scale("um", "mm") == pytest.approx(1e-3)
    assert file_unit_scale("mm", "mm") == pytest.approx(1.0)


def test_polygon_accepts_numpy_vertices() -> None:
    """Issue 4748 bug 3: get_polygons_points() returns Nx2 numpy arrays."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    text = polygon(verts, decimal_digits=4)
    assert text.startswith("G36*")
    assert "X0Y0D02*" in text
    assert "X10000Y0D01*" in text
    assert text.endswith("X0Y0D01*\nG37*\n\n")


def test_decimal_digits_rejects_unknown_resolution() -> None:
    with pytest.raises(ValueError, match="Unsupported Gerber resolution"):
        decimal_digits(1e-9)


def test_to_gerber_writes_um_layout_as_mm_gerber(tmp_path: Path) -> None:
    """10 um x 5 um rectangle becomes 0.01 mm x 0.005 mm with a closed region."""
    written = to_gerber(
        _rectangle(),
        tmp_path,
        {WG: COPPER},
        options=GerberOptions(resolution=1e-6, int_size=4),
    )
    path = tmp_path / "F_Cu.gbr"
    assert path in written
    text = path.read_text()

    assert "%FSLAX46Y46*%" in text
    assert "%FSLA46Y46X*%" not in text
    assert "%TF.GenerationSoftware,gdsfactory,gdsfactory," in text
    assert "%MOMM*%" in text
    # 10 um -> 0.01 mm at 6 decimals -> X10000
    assert "X0Y0D02*" in text
    assert "X10000Y0D01*" in text
    assert "X10000Y5000D01*" in text
    assert "X0Y5000D01*" in text
    assert text.index("X0Y0D02*") < text.rindex("X0Y0D01*")
    assert text.rstrip().endswith("M02*")


def test_to_gerber_uses_tuple_layer_keys(tmp_path: Path) -> None:
    """Issue 4748 bug 1: default by='index' misses tuple layermap keys."""
    component = _rectangle()
    by_index = component.get_polygons_points()
    by_tuple = component.get_polygons_points(by="tuple")
    assert WG not in by_index
    assert WG in by_tuple

    to_gerber(component, tmp_path, {WG: COPPER}, write_job=False)
    text = (tmp_path / "F_Cu.gbr").read_text()
    assert "G36*" in text, (
        "export wrote an empty Gerber because layer keys did not match"
    )


def test_to_gerber_writes_ucamco_job_file(tmp_path: Path) -> None:
    """BoardOptions is no longer a stub: CAD must ship size + layer count."""
    written = to_gerber(
        _rectangle(),
        tmp_path,
        {WG: COPPER},
        board=BoardOptions(n_layers=1),
    )
    job_path = next(path for path in written if path.suffix == ".gbrjob")
    job = json.loads(job_path.read_text())
    assert job["Header"]["GenerationSoftware"]["Vendor"] == "gdsfactory"
    assert job["GeneralSpecs"]["Size"]["X"] == pytest.approx(0.01)
    assert job["GeneralSpecs"]["Size"]["Y"] == pytest.approx(0.005)
    assert job["GeneralSpecs"]["LayerNumber"] == 1
    assert job["FilesAttributes"][0]["FileFunction"] == "Copper,L1,Top"
    assert job["FilesAttributes"][0]["Path"] == "F_Cu.gbr"


def test_to_gerber_pad_mtop_layer_enum(tmp_path: Path) -> None:
    """RF pad on MTOP: LayerEnum keys resolve, geometry is in millimetres."""
    pad = gf.components.pad(size=(100, 100))
    written = to_gerber(
        pad,
        tmp_path,
        {
            LAYER.MTOP: GerberLayer(
                name="F_Cu",
                function=["Copper", "L1", "Top"],
                polarity="Positive",
            )
        },
    )
    text = (tmp_path / "F_Cu.gbr").read_text()
    assert "G36*" in text
    assert any(path.suffix == ".gbrjob" for path in written)
    job = json.loads(
        next(path for path in written if path.suffix == ".gbrjob").read_text()
    )
    assert job["GeneralSpecs"]["Size"]["X"] == pytest.approx(0.1)
    assert job["GeneralSpecs"]["Size"]["Y"] == pytest.approx(0.1)
