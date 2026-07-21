from __future__ import annotations

import json
import warnings
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


def test_import_gds_cross_section_naming_conflict(tmp_path: Path) -> None:
    """Import a GDS whose port cross-section matches an existing one under a different name.

    Ports are serialized by cross-section name, so an older GDS may reference a
    structurally-identical cross-section under a legacy name. On import that name
    must resolve to the already-registered (canonical) cross-section instead of
    raising ``CrossSectionNamingConflictError``/``KeyError``.
    """
    gf.gpdk.PDK.activate()

    width_dbu = 602  # even (dbu symmetry) and unlikely to collide with defaults
    layer_info = kf.kdb.LayerInfo(2, 0)
    legacy_name = "strip_legacy_conflict"
    canonical_name = "strip_canonical_conflict"

    # Build an isolated layout whose top cell has a port using ``legacy_name``,
    # then write it to a GDS. Using a separate KCLayout keeps the file's name out
    # of the global registry so the conflict only arises at import time.
    file_kcl = kf.KCLayout("legacy_file_layout_conflict")
    enclosure = file_kcl.get_enclosure(
        kf.LayerEnclosure(sections=[], main_layer=layer_info)
    )
    xs_legacy = file_kcl.get_symmetrical_cross_section(
        kf.SymmetricalCrossSection(
            width=width_dbu, enclosure=enclosure, name=legacy_name
        )
    )
    cell = file_kcl.kcell("TOP_LEGACY_CONFLICT")
    cell.shapes(file_kcl.layout.layer(layer_info)).insert(
        kf.kdb.Box(0, -width_dbu // 2, 10000, width_dbu // 2)
    )
    cell.create_port(
        name="o1",
        trans=kf.kdb.Trans(0, False, 0, 0),
        cross_section=xs_legacy,
        port_type="optical",
    )
    cell.set_meta_data()
    gdspath = tmp_path / "legacy.gds"
    file_kcl.write(str(gdspath))

    registry = kf.kcl.cross_sections.cross_sections
    keys_before = set(registry)
    try:
        # Register the same structure globally under a different (canonical) name.
        kf.kcl.get_symmetrical_cross_section(
            kf.SymmetricalCrossSection(
                width=width_dbu,
                enclosure=kf.kcl.get_enclosure(
                    kf.LayerEnclosure(sections=[], main_layer=layer_info)
                ),
                name=canonical_name,
            )
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            c = import_gds(gdspath)
            messages = [str(w.message) for w in caught]

        assert len(c.ports) == 1
        # The imported port resolves to the canonical cross-section, not the
        # legacy name it was written with.
        assert c.ports[0].info["cross_section"] == canonical_name
        assert any(legacy_name in m and canonical_name in m for m in messages), messages
    finally:
        for key in set(registry) - keys_before:
            del registry[key]
        file_kcl.library.delete()
        kf.layout.kcls.pop(file_kcl.name, None)


def test_import_gds_cross_section_radius_conflict(tmp_path: Path) -> None:
    """A genuine radius conflict must still surface instead of being silently aliased."""
    gf.gpdk.PDK.activate()

    width_dbu = 604
    layer_info = kf.kdb.LayerInfo(3, 0)

    file_kcl = kf.KCLayout("legacy_file_layout_radius")
    enclosure = file_kcl.get_enclosure(
        kf.LayerEnclosure(sections=[], main_layer=layer_info)
    )
    xs_legacy = file_kcl.get_symmetrical_cross_section(
        kf.SymmetricalCrossSection(
            width=width_dbu,
            enclosure=enclosure,
            name="strip_radius_legacy",
            radius=5000,
        )
    )
    cell = file_kcl.kcell("TOP_RADIUS")
    cell.shapes(file_kcl.layout.layer(layer_info)).insert(
        kf.kdb.Box(0, -width_dbu // 2, 10000, width_dbu // 2)
    )
    cell.create_port(
        name="o1",
        trans=kf.kdb.Trans(0, False, 0, 0),
        cross_section=xs_legacy,
        port_type="optical",
    )
    cell.set_meta_data()
    gdspath = tmp_path / "legacy_radius.gds"
    file_kcl.write(str(gdspath))

    registry = kf.kcl.cross_sections.cross_sections
    keys_before = set(registry)
    try:
        # Same structure, but a different (incompatible) radius: must not be aliased.
        kf.kcl.get_symmetrical_cross_section(
            kf.SymmetricalCrossSection(
                width=width_dbu,
                enclosure=kf.kcl.get_enclosure(
                    kf.LayerEnclosure(sections=[], main_layer=layer_info)
                ),
                name="strip_radius_canonical",
                radius=10000,
            )
        )
        with pytest.raises(kf.exceptions.CrossSectionNamingConflictError):
            import_gds(gdspath)
    finally:
        for key in set(registry) - keys_before:
            del registry[key]
        file_kcl.library.delete()
        kf.layout.kcls.pop(file_kcl.name, None)


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
