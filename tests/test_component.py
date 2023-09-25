import pytest

import gdsfactory as gf


def test_same_uid() -> None:
    c = gf.Component()
    _ = c << gf.components.rectangle()
    _ = c << gf.components.rectangle()

    r1 = c.references[0].parent
    r2 = c.references[1].parent

    assert r1.uid == r2.uid, f"{r1.uid} must equal {r2.uid}"


def test_netlist_simple() -> None:
    c = gf.Component()
    c1 = c << gf.components.straight(length=1, width=2)
    c2 = c << gf.components.straight(length=2, width=2)
    c2.connect(port="o1", destination=c1.ports["o2"])
    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    netlist = c.get_netlist()
    assert len(netlist["instances"]) == 2


def test_netlist_simple_width_mismatch_throws_error() -> None:
    c = gf.Component()
    c1 = c << gf.components.straight(length=1, width=1)
    c2 = c << gf.components.straight(length=2, width=2)
    c2.connect(port="o1", destination=c1.ports["o2"])
    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    with pytest.raises(ValueError):
        c.get_netlist()


def test_netlist_complex() -> None:
    c = gf.components.mzi_arms()
    netlist = c.get_netlist()
    # print(netlist.pretty())
    assert len(netlist["instances"]) == 4, len(netlist["instances"])


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


def test_import_gds_settings() -> None:
    c = gf.components.mzi()
    gdspath = c.write_gds(with_metadata=True)
    c2 = gf.import_gds(gdspath, name="mzi_sample", read_metadata=True)
    c3 = gf.routing.add_fiber_single(c2)
    assert c3


if __name__ == "__main__":
    test_extract()
