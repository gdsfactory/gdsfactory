from __future__ import annotations

import json

import jsondiff  # type: ignore
import pandas as pd  # type: ignore
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.generic_tech.layer_map import LAYER
from gdsfactory.read.import_gds import import_gds
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
        gf.components.compass(layer="WG"), rows=2, columns=2, spacing=(10, 10)
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


if __name__ == "__main__":
    test_import_gds_array()
