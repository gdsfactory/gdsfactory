import gdsfactory as gf
from gdsfactory.import_gds import import_gds


def test_import_gds_snap_to_grid() -> None:
    gdspath = gf.CONFIG["gdsdir"] / "mmi1x2.gds"
    c = import_gds(gdspath, snap_to_grid_nm=5)
    assert len(c.get_polygons()) == 8, len(c.get_polygons())

    for x, y in c.get_polygons()[0]:
        assert gf.snap.is_on_grid(x, 5)
        assert gf.snap.is_on_grid(y, 5)


def test_import_gds_hierarchy() -> None:
    c0 = gf.components.mzi()
    gdspath = c0.write_gds()
    c = import_gds(gdspath)
    assert len(c.get_dependencies()) == 3, len(c.get_dependencies())


if __name__ == "__main__":
    test_import_gds_hierarchy()
