import pp
from pp.import_gds import import_gds


def test_import_gds_snap_to_grid() -> None:
    gdspath = pp.CONFIG["gdsdir"] / "mmi1x2.gds"
    c = import_gds(gdspath, snap_to_grid_nm=5)
    print(len(c.get_polygons()))
    assert len(c.get_polygons()) == 8

    for x, y in c.get_polygons()[0]:
        assert pp.snap.is_on_grid(x, 5)
        assert pp.snap.is_on_grid(y, 5)


def test_import_gds_hierarchy() -> None:
    c0 = pp.components.mzi2x2()
    gdspath = c0.write_gds()
    c = import_gds(gdspath)
    assert len(c.get_dependencies()) == 3


if __name__ == "__main__":
    test_import_gds_hierarchy()
