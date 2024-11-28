import kfactory as kf

import gdsfactory as gf
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
    c1_area = c1.area(layer=layer)

    c1.trim(left=-5, right=5, top=5, bottom=-5)
    assert c1_area == c1.area(layer=layer), f"{c1_area} != {c1.area(layer=layer)}"


def test_from_kcell() -> None:
    kf.kcl.infos = kf.LayerInfos(WG=kf.kdb.LayerInfo(1, 0))
    gf.Component.from_kcell(kf.cells.straight.straight(1, 1, gf.kcl.get_info(LAYER.WG)))


def test_remove_layers_recursive() -> None:
    comp = gf.Component()
    r1 = gf.components.rectangle(size=(1, 15), layer=(1, 0), centered=True)
    _ = comp << r1
    r2 = gf.components.rectangle(size=(2, 30), layer=(2, 0), centered=True)
    _ = comp << r2
    comp.flatten()

    copy = comp.dup()
    copy.remove_layers(layers=[(2, 0)])

    assert comp.area((1, 0)) == 15, f"{comp.area((1, 0))}"
    assert comp.area((2, 0)) == 60, f"{comp.area((2, 0))}"


def test_remove_layers_flat() -> None:
    comp = gf.Component()
    comp.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(2, 0))

    copy = comp.dup()
    copy.remove_layers(layers=[(2, 0)])
    assert comp.area((2, 0)) == 0, f"{comp.area((2, 0))}"
    assert copy.area((2, 0)) == 100, f"{copy.area((2, 0))}"
