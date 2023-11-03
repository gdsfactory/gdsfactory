import gdsfactory as gf


def test_component_copy_two_copies_in_one() -> None:
    c = gf.Component()
    c1 = gf.components.straight()
    c2 = c1.copy()
    c3 = c1.copy()
    c3.add_label("I'm different")

    _ = c << c1
    r2 = c << c2
    r3 = c << c3
    r2.movey(-100)
    r3.movey(-200)
    assert c2.name != c3.name


def test_component_copy() -> None:
    c1 = gf.components.straight()
    c2 = c1.copy()
    assert (
        len(c1.info) > 0
    ), "This test doesn't make any sense unless there is some info to copy"
    assert c1.info == c2.info


def test_extract() -> None:
    WGCLAD = (111, 0)

    xs = gf.cross_section.strip(
        width=0.5,
        bbox_layers=[WGCLAD],
        bbox_offsets=[3],
        cladding_layers=None,
    )

    c = gf.components.straight(
        length=10,
        cross_section=xs,
        add_pins=False,
    )
    c2 = c.extract(layers=[WGCLAD])

    assert len(c.polygons) == 2, len(c.polygons)
    assert len(c2.polygons) == 1, len(c2.polygons)
    assert WGCLAD in c2.layers


def test_bbox_snap_to_grid() -> None:
    c = gf.Component()
    c1 = c << gf.components.rectangle(size=(1.5e-3, 1.5e-3), port_type=None)
    c2 = c << gf.components.rectangle(size=(1.5e-3, 1.5e-3), port_type=None)
    c2.xmin = c1.xmax

    assert c2.xsize == 0.002, c2.xsize


def test_remove_labels() -> None:
    c = gf.c.straight()
    c.remove_labels()

    assert len(c.labels) == 0


if __name__ == "__main__":
    test_extract()
