import gdsfactory as gf
from gdsfactory.add_ports import add_ports_from_markers_inside
from gdsfactory.read.import_gds import import_gds


def test_import_gds_snap_to_grid() -> None:
    gdspath = gf.CONFIG["gdsdir"] / "mmi1x2.gds"
    c = import_gds(gdspath, snap_to_grid_nm=5)
    assert len(c.get_polygons()) == 8, len(c.get_polygons())

    for x, y in c.get_polygons()[0]:
        assert gf.snap.is_on_grid(x, 5)
        assert gf.snap.is_on_grid(y, 5)


def test_import_gds_hierarchy() -> gf.Component:
    c0 = gf.components.mzi_arms()
    gdspath = c0.write_gds()
    gf.clear_cache()

    c = import_gds(gdspath)
    assert len(c.get_dependencies()) == 2, len(c.get_dependencies())
    assert c.name == c0.name, c.name
    return c


def test_import_ports() -> gf.Component:
    """Make sure you can import the ports"""
    c0 = gf.components.mzi_arms(decorator=gf.add_pins)
    gdspath = c0.write_gds()
    c0x1 = c0.ports["o1"].x
    c0x2 = c0.ports["o2"].x
    gf.clear_cache()

    c1 = import_gds(gdspath, decorator=add_ports_from_markers_inside)
    c1x1 = c1.ports["o1"].x
    c1x2 = c1.ports["o2"].x

    assert c0x1 == c1x1
    assert c0x2 == c1x2
    return c1


def test_import_gds_add_padding() -> gf.Component:
    """Make sure you can import the ports"""
    c0 = gf.components.mzi_arms(decorator=gf.add_pins)
    gdspath = c0.write_gds()
    gf.clear_cache()

    c1 = import_gds(gdspath, decorator=gf.add_padding_container, name="mzi")
    assert c1.name == "mzi"
    return c1


if __name__ == "__main__":
    c = test_import_gds_hierarchy()
    # c = test_import_ports()
    # c = test_import_gds_add_padding()
    c.show()
    # test_import_gds_snap_to_grid()
