from __future__ import annotations

import json

import jsondiff
import pandas as pd

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER
from gdsfactory.read.import_gds import import_gds


def test_import_gds_info() -> None:
    """Ensures Component from GDS + YAML loads same component settings."""
    c1 = gf.components.straight(length=1.234)
    gdspath = gf.PATH.gdsdir / "straight.gds"

    c2 = gf.import_gds(gdspath)
    d1 = c1.to_dict()
    d2 = c2.to_dict()
    d = jsondiff.diff(d1, d2)
    assert len(d) == 0, d


def test_import_gds_hierarchy() -> None:
    c0 = gf.components.mzi_arms(delta_length=11)
    gdspath = c0.write_gds()

    c = import_gds(gdspath)
    assert c.name == c0.name, c.name


def test_import_json_label(data_regression) -> None:
    """Make sure you can import the ports."""
    c = gf.components.straight()
    c1 = gf.labels.add_label_json(c)
    gdspath = c1.write_gds()
    csvpath = gf.labels.write_labels(gdspath, prefix="{")

    df = pd.read_csv(csvpath)
    settings = json.loads(df.columns[0])
    data_regression.check(settings)


def test_import_gds_array() -> None:
    """Make sure you can import a GDS with arrays."""
    c0 = gf.components.array(
        gf.components.rectangle(layer=LAYER.WG), rows=2, columns=2, spacing=(10, 10)
    )
    gdspath = c0.write_gds()

    c1 = import_gds(gdspath)
    assert len(c1.get_polygons()[LAYER.WG]) == 4, len(c1.get_polygons()[LAYER.WG])


def test_import_gds_ports(data_regression) -> None:
    """Make sure you can import the ports."""
    c0 = gf.components.straight()
    gdspath = c0.write_gds()

    c1 = import_gds(
        gdspath,
    )
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
    if data_regression:
        data_regression.check(c1.to_dict())


if __name__ == "__main__":
    # test_import_gds_info()
    c1 = gf.components.straight(length=1.234)
    gdspath = gf.PATH.gdsdir / "straight.gds"

    c2 = gf.import_gds(gdspath)
    d1 = c1.to_dict()
    d2 = c2.to_dict()
    d = jsondiff.diff(d1, d2)
    assert len(d) == 0, d
