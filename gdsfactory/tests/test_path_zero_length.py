import gdsfactory as gf


def test_path_less_than_1nm() -> None:
    c = gf.components.straight(
        length=0.5e-3, cross_section=gf.cross_section.cross_section
    )
    assert not c.references
    assert not c.polygons


if __name__ == "__main__":
    test_path_less_than_1nm()
    c = gf.components.straight(length=0.5e-3)
    c.show(show_ports=True)
