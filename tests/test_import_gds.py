from __future__ import annotations

import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.read.import_gds import import_gds


@gf.cell(autoname=False, copy_if_cached=False)
def import_gds_custom(gdspath, **kwargs):
    return gf.import_gds(gdspath, **kwargs)


# def test_import_gds_snap_to_grid() -> None:
#     gdspath = gf.PATH.gdsdir / "mmi1x2.gds"
#     c = import_gds(gdspath, snap_to_grid_nm=5)
#     assert len(c.get_polygons()) == 8, len(c.get_polygons())

#     for polygon in c.get_polygons(by_spec=False):
#         assert gf.snap.is_on_grid(
#             polygon.points, 5
#         ), f"{polygon.points} not in 5nm grid"


def test_import_gds_hierarchy() -> None:
    c0 = gf.components.mzi_arms(delta_length=11)
    gdspath = c0.write_gds()

    c = import_gds(gdspath, unique_names=False)
    assert len(c.get_dependencies()) == 3, len(c.get_dependencies())
    assert c.name == c0.name, c.name


def test_import_gds_name() -> None:
    cellname = "thermal_phase_shifter_multimode_500um"
    c = import_gds(PATH.thermal, cellname=cellname)
    assert c.name == cellname, c.name


def test_import_gds_name_custom() -> None:
    cellname = "thermal_phase_shifter_multimode_500um"
    c = import_gds_custom(PATH.thermal, cellname=cellname)
    assert c.name == cellname, c.name


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
