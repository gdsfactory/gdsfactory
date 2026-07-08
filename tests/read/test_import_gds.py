from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import jsondiff
import kfactory as kf
import pandas as pd
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.gpdk.layer_map import LAYER
from gdsfactory.read.import_gds import import_gds, import_gds_multiple_top_cells
from gdsfactory.serialization import clean_value_json


def test_import_gds_info() -> None:
    """Ensures Component from GDS + YAML loads same component settings."""
    c1 = gf.components.straight(length=1.234)
    gdspath = c1.write_gds()

    c2 = gf.import_gds(gdspath)
    d1 = c1.to_dict()
    d2 = c2.to_dict()
    d = jsondiff.diff(d1, d2)
    assert len(d) == 0, d


def test_import_gds_hierarchy() -> None:
    """Import a GDS with hierarchy."""
    c0 = gf.components.mzi(delta_length=11)
    gdspath = c0.write_gds()

    c = import_gds(gdspath)
    assert c.name == c0.name, c.name


@pytest.mark.parametrize("reader", [import_gds, import_gds_multiple_top_cells])
def test_import_gds_cleans_up_temp_kcl_on_read_error(
    tmp_path: Path, reader: Callable[..., object]
) -> None:
    gdspath = tmp_path / "missing.gds"

    with pytest.raises(RuntimeError):
        reader(gdspath)

    assert str(gdspath) not in kf.layout.kcls


@pytest.mark.parametrize("reader", [import_gds, import_gds_multiple_top_cells])
def test_import_gds_cleans_up_temp_kcl_on_post_process_error(
    tmp_path: Path, reader: Callable[..., object]
) -> None:
    gdspath = gf.components.straight().write_gds(tmp_path / "straight.gds")

    def fail_post_process(component: gf.Component) -> None:
        raise RuntimeError("post process failed")

    with pytest.raises(RuntimeError, match="post process failed"):
        reader(gdspath, post_process=(fail_post_process,))

    assert str(gdspath) not in kf.layout.kcls


def test_import_gds_multiple_top_cells_cleans_up_temp_kcl_on_cellname_error(
    tmp_path: Path,
) -> None:
    gdspath = gf.components.straight().write_gds(tmp_path / "straight.gds")

    with pytest.raises(ValueError, match="Unknown cellnames requested"):
        import_gds_multiple_top_cells(gdspath, cellnames=["missing"])

    assert str(gdspath) not in kf.layout.kcls


def test_import_json_label(data_regression: DataRegressionFixture) -> None:
    """Import ports from GDS."""
    c = gf.components.straight().dup()
    c.name = "straight__test_import_json_label"
    c1 = gf.labels.add_label_json(c)
    gdspath = c1.write_gds()
    csvpath = gf.labels.write_labels(gdspath, prefixes=["{"])

    df = pd.read_csv(csvpath)
    settings = json.loads(df.iloc[0].text)
    if data_regression:
        settings = clean_value_json(settings)
        data_regression.check(settings)


def test_import_gds_array() -> None:
    """Import a GDS with InstanceArray."""
    c0 = gf.components.array(
        gf.components.compass(layer="WG"),
        rows=2,
        columns=2,
        column_pitch=10,
        row_pitch=10,
    )
    gdspath = c0.write_gds()

    c1 = import_gds(gdspath)
    assert len(c1.get_polygons()[LAYER.WG]) == 4, len(c1.get_polygons()[LAYER.WG])


def test_import_gds_ports(data_regression: DataRegressionFixture) -> None:
    """Import the ports."""
    c0 = gf.components.straight()
    gdspath = c0.write_gds()

    c1 = import_gds(
        gdspath,
    )
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
    if data_regression:
        data_regression.check(c1.to_dict())


def test_import_gds_subcell(data_regression: DataRegressionFixture) -> None:
    c0 = gf.Component()
    c1 = gf.components.mzi()
    c0.add_ref(c1, name="mzi")

    gdspath = c0.write_gds()

    l0 = gf.import_gds(gdspath)
    l1 = l0.insts["mzi"].cell

    assert len(l1.ports) == 2
    assert l1.settings == c1.settings


def import_same_file_twice() -> None:
    c1 = gf.c.straight()
    gdspath = c1.write_gds()

    c2 = gf.import_gds(gdspath)
    c3 = gf.import_gds(gdspath)

    c = gf.Component()
    c.add_ref(c2)
    c.add_ref(c3)
    c.write_gds()
    assert c
