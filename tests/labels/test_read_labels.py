from __future__ import annotations

from pathlib import Path

import klayout.db as kdb
import pytest

import gdsfactory as gf
from gdsfactory.gpdk.layer_map import LAYER
from gdsfactory.typings import LayerSpec


def test_write_labels() -> None:
    """Write labels to a CSV file."""
    c = gf.c.straight()
    c = gf.add_pins.add_pins_siepic_container(c)

    gdspath = c.write_gds()
    labels = list(
        gf.labels.find_labels(gdspath, layer_label=LAYER.PORT, prefixes=["o"])
    )
    gf.labels.write_labels(gdspath, layer_label=LAYER.PORT, prefixes=["o"])
    assert len(labels) == 2


@pytest.fixture
def label_gds(tmp_path: Path) -> Path:
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    child = layout.create_cell("CHILD")
    layer = layout.layer(*gf.get_layer_tuple("PORT"))
    child.shapes(layer).insert(kdb.Text("opt-device", kdb.Trans(1000, 2000)))
    child.shapes(layer).insert(kdb.Text("unmatched", kdb.Trans()))
    child.shapes(layer).insert(kdb.Box(0, 0, 3000, 3000))
    top.insert(kdb.CellInstArray(child.cell_index(), kdb.Trans(1, False, 10000, 20000)))
    top.shapes(layout.layer(999, 0)).insert(kdb.Text("opt-other-layer", kdb.Trans()))
    gdspath = tmp_path / "labels.gds"
    layout.write(str(gdspath))
    return gdspath


@pytest.mark.parametrize("layer_label", ["PORT", (1, 10), LAYER.PORT])
def test_find_labels_layer_specs(label_gds: Path, layer_label: LayerSpec) -> None:
    labels = list(
        gf.labels.find_labels(
            label_gds, layer_label=layer_label, prefixes=("opt-", "opt-device")
        )
    )
    assert labels == [("opt-device", 8.0, 21.0, 90.0)] * 2


def test_find_labels_invalid_layer(label_gds: Path) -> None:
    with pytest.raises(ValueError, match=r"unknown_label_layer.*not in PDK"):
        list(gf.labels.find_labels(label_gds, layer_label="unknown_label_layer"))


def test_find_labels_read_error_before_invalid_layer(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Unable to open file"):
        list(
            gf.labels.find_labels(
                tmp_path / "missing.gds", layer_label="unknown_label_layer"
            )
        )
