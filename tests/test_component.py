import kfactory as kf
import pytest

import gdsfactory as gf
from gdsfactory.component import LockedError
from gdsfactory.generic_tech import LAYER


def test_component_dup() -> None:
    c1 = gf.components.straight(length=10)
    c2 = c1.dup()
    assert c1.settings.length == 10
    assert c1.info == c2.info


def test_extract() -> None:
    xs = gf.cross_section.strip(
        width=0.5,
        bbox_layers=(LAYER.WGCLAD,),
        bbox_offsets=(3,),
    )

    c = gf.components.straight(
        length=11.124,
        cross_section=xs,
    )
    c2 = c.extract(layers=[LAYER.WGCLAD])
    p = 1
    c2_polygons = c2.get_polygons()
    assert len(c2_polygons) == p, len(c2_polygons)
    assert tuple(LAYER.WGCLAD) in c2.layers, c2.layers


def test_hierarchy() -> None:
    c = gf.c.mzi()
    assert len(c.called_cells()) == 5, len(c.called_cells())
    assert c.child_cells() == 5, c.child_cells()


def test_get_polygons() -> None:
    c = gf.components.straight()
    polygons = c.get_polygons(by="index")
    assert 1 in polygons

    polygons = c.get_polygons(by="name")
    assert "WG" in polygons

    polygons = c.get_polygons(by="tuple")
    assert (1, 0) in polygons


def test_trim() -> None:
    layer = (1, 0)
    c1 = gf.c.rectangle(size=(9, 9), centered=True, layer=layer).dup()
    c1.flatten()
    c1_area = c1.area(layer=layer)

    c1.trim(left=-5, right=5, top=5, bottom=-5)
    assert c1_area == c1.area(layer=layer), f"{c1_area} != {c1.area(layer=layer)}"


def test_from_kcell() -> None:
    kf.kcl.infos = kf.LayerInfos(WG=kf.kdb.LayerInfo(1, 0))
    gf.Component.from_kcell(kf.cells.straight.straight(1, 1, gf.kcl.get_info(LAYER.WG)))


def test_remove_layers_recursive() -> None:
    comp = gf.Component()
    r1 = gf.components.compass(size=(1, 15), layer=(1, 0))
    _ = comp << r1
    r2 = gf.components.compass(size=(2, 30), layer=(2, 0))
    _ = comp << r2
    comp.flatten()

    copy = comp.dup()
    copy.remove_layers(layers=[(2, 0)])

    assert comp.area((1, 0)) == 15, f"{comp.area((1, 0))}"
    assert comp.area((2, 0)) == 60, f"{comp.area((2, 0))}"


def test_remove_layers_flat() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(2, 0))

    empty = c.dup()
    empty.remove_layers(layers=[(2, 0)])
    assert c.area((2, 0)) == 100, f"{c.area((2, 0))}"
    assert empty.area((2, 0)) == 0, f"{empty.area((2, 0))}"


def test_locked_cell() -> None:
    c = gf.components.straight()

    with pytest.raises(LockedError):
        c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(2, 0))

    with pytest.raises(LockedError):
        c.remove_layers(layers=["WG"])

    with pytest.raises(LockedError):
        c.remap_layers({"WG": "SLAB90"})

    with pytest.raises(LockedError):
        c.copy_layers({"WG": "SLAB90"})

    with pytest.raises(LockedError):
        c.over_under("WG")

    with pytest.raises(LockedError):
        c.offset("WG", distance=0.1)

    with pytest.raises(LockedError):
        c.add_port(name="o1", center=(0, 0), width=0.5, orientation=0, layer="WG")
