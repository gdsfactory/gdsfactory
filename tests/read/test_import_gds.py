from __future__ import annotations

import jsondiff

import gdsfactory as gf
from gdsfactory.read.import_gds import import_gds

# def test_import_gds_snap_to_grid() -> None:
#     gdspath = gf.PATH.gdsdir / "mmi1x2.gds"
#     c = import_gds(gdspath, snap_to_grid_nm=5)
#     assert len(c.get_polygons()) == 8, len(c.get_polygons())

#     for polygon in c.get_polygons(by_spec=False):
#         assert gf.snap.is_on_grid(
#             polygon.points, 5
#         ), f"{polygon.points} not in 5nm grid"


def test_read_gds_equivalent() -> None:
    """Ensures Component from GDS + YAML loads same component settings."""
    c1 = gf.components.straight(length=1.234)
    gdspath = gf.PATH.gdsdir / "straight.gds"

    c2 = gf.import_gds(gdspath, read_metadata=True, unique_names=False)
    d1 = c1.to_dict()
    d2 = c2.to_dict()
    d1.pop("name")
    d2.pop("name")
    d = jsondiff.diff(d1, d2)

    assert len(d) == 0, d


def test_import_gds_hierarchy() -> None:
    c0 = gf.components.mzi_arms(delta_length=11)
    gdspath = c0.write_gds()

    c = import_gds(gdspath, unique_names=False)
    assert len(c.insts) == 3, len(c.insts)
    assert c.name == c0.name, c.name


# def test_import_gds_add_padding() -> None:
#     """Make sure you can import the ports"""
#     c0 = gf.components.mzi_arms(decorator=gf.add_pins)
#     gdspath = c0.write_gds()

#     c1 = import_gds(gdspath, decorator=gf.add_padding_container, name="mzi")
#     assert c1.name == "mzi"


def test_import_gds_array() -> None:
    """Make sure you can import a GDS with arrays."""
    c0 = gf.components.array(
        gf.components.rectangle, rows=2, columns=2, spacing=(10, 10)
    )
    gdspath = c0.write_gds()

    c1 = import_gds(gdspath)
    assert len(c1.get_polygons()) == 4


def test_import_gds_raw() -> None:
    """Make sure you can import a GDS with arrays."""
    c0 = gf.components.array(
        gf.components.rectangle, rows=2, columns=2, spacing=(10, 10)
    )
    gdspath = c0.write_gds()

    c = gf.read.import_gds(gdspath)
    assert c
