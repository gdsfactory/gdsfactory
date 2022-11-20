import gdsfactory as gf
from gdsfactory.read.import_gds import import_gds

# def test_import_gds_snap_to_grid() -> None:
#     gdspath = gf.CONFIG["gdsdir"] / "mmi1x2.gds"
#     c = import_gds(gdspath, snap_to_grid_nm=5)
#     assert len(c.get_polygons()) == 8, len(c.get_polygons())

#     for polygon in c.get_polygons(by_spec=False):
#         assert gf.snap.is_on_grid(
#             polygon.points, 5
#         ), f"{polygon.points} not in 5nm grid"


def test_import_gds_hierarchy() -> gf.Component:
    c0 = gf.components.mzi_arms(delta_length=11)
    gdspath = c0.write_gds()

    c = import_gds(gdspath)
    assert len(c.get_dependencies()) == 3, len(c.get_dependencies())
    assert c.name == c0.name, c.name
    return c


def test_import_ports_inside() -> gf.Component:
    """Make sure you can import the ports"""
    c0 = gf.components.straight(decorator=gf.add_pins.add_pins)
    gdspath = c0.write_gds()

    gf.clear_cache()
    c1 = import_gds(gdspath, decorator=gf.add_ports.add_ports_from_markers_inside)
    assert len(c1.ports) == 2
    return c1


# def test_import_gds_add_padding() -> gf.Component:
#     """Make sure you can import the ports"""
#     c0 = gf.components.mzi_arms(decorator=gf.add_pins)
#     gdspath = c0.write_gds()
#     gf.clear_cache()

#     c1 = import_gds(gdspath, decorator=gf.add_padding_container, name="mzi")
#     assert c1.name == "mzi"
#     return c1


if __name__ == "__main__":
    # c = test_import_gds_hierarchy()
    c = test_import_ports_inside()
    c.show()

    # c = test_import_ports()
    # c = test_import_gds_add_padding()
    # c.show(show_ports=True)
    # test_import_gds_snap_to_grid()

    # cross_section = gf.cross_section.cross_section
    # splitter = gf.components.mmi1x2(cross_section=cross_section)
    # c0 = gf.components.mzi_arms(splitter=splitter, cross_section=cross_section)
    # c0.unlock()
    # c0 = add_pins(c0)
    # c0.lock()

    # gdspath = c0.write_gds()
    # c0x1 = c0.ports["o1"].x
    # c0x2 = c0.ports["o2"].x

    # c1 = import_gds(gdspath, decorator=add_ports_from_markers_inside)
    # c1x1 = c1.ports["o1"].x
    # c1x2 = c1.ports["o2"].x

    # assert c0x1 == c1x1
    # assert c0x2 == c1x2

    # c1.show(show_ports=True)
