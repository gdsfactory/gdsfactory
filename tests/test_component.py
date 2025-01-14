from typing import Any

import kfactory as kf
import klayout.db as kdb
import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory.component import (
    ComponentReference,
    LockedError,
    component_with_function,
    container,
    copy,
    ensure_tuple_of_tuples,
    points_to_polygon,
    size,
)
from gdsfactory.config import GDSDIR_TEMP
from gdsfactory.generic_tech import LAYER
from gdsfactory.pdk import get_layer


def test_component_copy() -> None:
    c1 = gf.components.straight(length=10)
    c2 = c1.dup()
    assert c1.settings["length"] == 10
    assert c1.info == c2.info


def test_component_all_angle_copy() -> None:
    c1 = gf.components.straight_all_angle(length=10)
    c2 = c1.dup()
    assert c1.settings["length"] == 10
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
    c1 = gf.c.rectangle(size=(9, 9), centered=True, layer=layer).copy()
    c1.flatten()
    c1_area = c1.area(layer=layer)

    c1.trim(left=-5, right=5, top=5, bottom=-5)
    assert c1_area == c1.area(layer=layer), f"{c1_area} != {c1.area(layer=layer)}"


def test_from_kcell() -> None:
    kf.kcl.infos = kf.LayerInfos(WG=kf.kdb.LayerInfo(1, 0))
    gf.Component.from_kcell(kf.cells.straight.straight(1, 1, gf.kcl.get_info(LAYER.WG)))


def test_remove_layers() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(2, 0))

    c.remove_layers(layers=[(2, 0)])
    assert c.area((1, 0)) == 100, f"{c.area((1, 0))}"
    assert c.area((2, 0)) == 0, f"{c.area((2, 0))}"


def test_locked_cell() -> None:
    c = gf.Component()
    c._locked = True

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


def test_locked_cell_all_angle() -> None:
    c = gf.ComponentAllAngle()
    c._locked = True

    with pytest.raises(LockedError):
        c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(2, 0))

    with pytest.raises(LockedError):
        c.add_port(name="o1", center=(0, 0), width=0.5, orientation=0, layer="WG")

    with pytest.raises(LockedError):
        c.add_label()

    with pytest.raises(LockedError):
        c.add_route_info("strip", length=10)

    with pytest.raises(LockedError):
        c.copy_child_info(gf.components.straight().copy())


def test_ensure_tuple_of_tuples() -> None:
    import numpy as np

    points_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result_np = ensure_tuple_of_tuples(points_np)
    assert isinstance(result_np, tuple)
    assert all(isinstance(p, tuple) for p in result_np)
    assert result_np == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))

    points_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    result_list = ensure_tuple_of_tuples(points_list)
    assert isinstance(result_list, tuple)
    assert all(isinstance(p, tuple) for p in result_list)
    assert result_list == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))

    points_np_list = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    result_np_list = ensure_tuple_of_tuples(points_np_list)
    assert isinstance(result_np_list, tuple)
    assert all(isinstance(p, tuple) for p in result_np_list)
    assert result_np_list == ((1.0, 2.0), (3.0, 4.0))

    points_tuple = ((1.0, 2.0), (3.0, 4.0))
    result_tuple = ensure_tuple_of_tuples(points_tuple)
    assert result_tuple == points_tuple


def test_points_to_polygon() -> None:
    import klayout.db as kdb
    import numpy as np

    points_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result_np = points_to_polygon(points_np)
    assert isinstance(result_np, kdb.DPolygon)

    points_tuple = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    result_tuple = points_to_polygon(points_tuple)
    assert isinstance(result_tuple, kdb.DPolygon)

    kdb_polygon = kdb.DPolygon(kdb.DBox(0, 0, 10, 10))
    result_kdb = points_to_polygon(kdb_polygon)
    assert result_kdb == kdb_polygon

    kdb_simple = kdb.DSimplePolygon(kdb.DBox(0, 0, 10, 10))
    result_simple = points_to_polygon(kdb_simple)
    assert isinstance(result_simple, kdb.DPolygon | kdb.DSimplePolygon)

    region = kdb.Region()
    region.insert(kdb.Box(0, 0, 1000, 1000))
    result_region = points_to_polygon(region)
    assert isinstance(result_region, kdb.Region | kdb.DPolygon)


def test_size() -> None:
    region = kdb.Region()
    region.insert(kdb.Box(0, 0, 1000, 1000))

    result = size(region, offset=0.1)
    assert result.bbox().width() > region.bbox().width()

    result = size(region, offset=-0.1)
    assert result.bbox().width() < region.bbox().width()

    result = size(region, offset=0.1, dbu=2000)
    assert result.bbox().width() > region.bbox().width()


def test_region_copy() -> None:
    region = kdb.Region()
    region.insert(kdb.Box(0, 0, 1000, 1000))
    region_copy = copy(region)
    assert region_copy.bbox().width() == region.bbox().width()


def test_component_reference_properties() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight

    assert ref.center == ref.dcenter
    assert ref.x == ref.dx
    assert ref.y == ref.dy
    assert ref.xmin == ref.dxmin
    assert ref.xmax == ref.dxmax
    assert ref.ymin == ref.dymin
    assert ref.ymax == ref.dymax
    assert ref.xsize == ref.dxsize
    assert ref.ysize == ref.dysize
    assert ref.size_info == ref.dsize_info


def test_component_reference_setters() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight

    ref.x = 5.0  # type: ignore
    assert ref.dx == 5.0
    ref.y = 10.0  # type: ignore
    assert ref.dy == 10.0


def test_component_reference_transformations() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight

    original_pos = ref.center
    ref.move((1.0, 2.0))  # type: ignore
    assert ref.center != original_pos

    ref.rotate(90)  # type: ignore
    ref.mirror()


def test_component_reference_name() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight

    ref.name = "test_instance"
    assert ref.name == "test_instance"


def test_component_reference_equality_and_hash() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight
    inst2 = c << straight
    ref2 = ComponentReference(inst2)

    assert ref != ref2
    assert ref == ref
    assert ref != ""
    hash(ref)


def test_component_reference_flatten() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight
    ref.flatten()


def test_component_reference_connect() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    c2 = gf.components.straight(length=10).copy()
    inst2 = c << c2
    port = straight.add_port(
        name="o1", center=(0, 0), width=0.5, orientation=0, layer=(1, 0)
    )
    ref = c << straight

    with pytest.warns(DeprecationWarning):
        ref.connect("o1", other=inst2, destination=port, other_port_name="o1")
    with pytest.warns(DeprecationWarning):
        ref.connect("o1", port, overlap=1.0)
    with pytest.warns(DeprecationWarning):
        ref.connect("o1", port, preserve_orientation=True)


def test_component_reference_deprecated_attributes() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight

    with pytest.warns(DeprecationWarning):
        info = ref.info
        assert isinstance(info, dict)

    with pytest.warns(DeprecationWarning):
        parent = ref.parent
        assert parent == ref.cell


def test_component_references_getitem() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight
    ref2 = c << straight
    ref2.name = "test_ref"

    assert c.insts[0] == ref
    assert c.insts[1] == ref2

    assert c.insts["test_ref"] == ref2


def test_component_references_iter() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight
    ref2 = c << straight

    refs = list(c.insts)
    assert len(refs) == 2
    assert refs[0] == ref
    assert refs[1] == ref2
    assert all(isinstance(r, ComponentReference) for r in refs)


def test_component_references_delitem() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    c << straight

    del c.insts[0]
    assert len(c.insts) == 0


def test_component_all_angle_add_port() -> None:
    c = gf.ComponentAllAngle()

    port1 = c.add_port(
        name="p1",
        center=(10, 20),
        width=0.5,
        orientation=90,
        layer="WG",
        port_type="optical",
    )
    assert port1.name == "p1"
    assert port1.dwidth == 0.5
    assert port1.orientation == 90
    assert port1.port_type == "optical"
    assert port1.dcenter == (10, 20)

    port2 = c.add_port(
        name="p2",
        center=kdb.DPoint(30, 40),
        width=1.0,
        orientation=0,
        layer="WG",
        port_type="electrical",
    )
    assert port2.name == "p2"
    assert port2.dwidth == 1.0
    assert port2.orientation == 0
    assert port2.port_type == "electrical"
    assert port2.dcenter == (30, 40)

    port3 = c.add_port(
        name="p3",
        center=(50, 60),
        orientation=180,
        cross_section="strip",
        port_type="optical",
    )
    assert port3.name == "p3"
    assert port3.orientation == 180
    assert port3.port_type == "optical"
    assert port3.dcenter == (50, 60)

    with pytest.raises(ValueError, match="Must specify orientation"):
        c.add_port(name="p4", center=(0, 0), width=0.5, layer="WG")

    with pytest.raises(ValueError, match="Must specify center"):
        c.add_port(name="p5", width=0.5, orientation=90, layer="WG")

    with pytest.raises(ValueError, match="Must specify layer or cross_section"):
        c.add_port(name="p6", center=(0, 0), width=0.5, orientation=90)

    with pytest.raises(ValueError, match="Must specify width or cross_section"):
        c.add_port(name="p7", center=(0, 0), orientation=90, layer="WG")

    with pytest.warns(UserWarning, match="Port type invalid not in"):
        c.add_port(
            name="p8",
            center=(0, 0),
            width=0.5,
            orientation=90,
            layer="WG",
            port_type="invalid",
        )


def test_component_all_angle_add_polygon() -> None:
    c = gf.ComponentAllAngle()

    points1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
    c.add_polygon(points1, layer="WG")
    assert len(list(c.shapes(get_layer(LAYER.WG)).each())) == 1

    points2 = np.array([(20, 0), (30, 0), (30, 10), (20, 10)])
    c.add_polygon(points2, layer="WG")
    assert len(list(c.shapes(get_layer(LAYER.WG)).each())) == 2

    points3 = [
        kf.kdb.DPoint(40, 0),
        kf.kdb.DPoint(50, 0),
        kf.kdb.DPoint(50, 10),
        kf.kdb.DPoint(40, 10),
    ]
    dpoly = kf.kdb.DPolygon()
    dpoly.assign_hull(points3)
    c.add_polygon(dpoly, layer="WG")
    assert len(list(c.shapes(get_layer(LAYER.WG)).each())) == 3
    points4 = [
        kf.kdb.Point(60, 0),
        kf.kdb.Point(70, 0),
        kf.kdb.Point(70, 10),
        kf.kdb.Point(60, 10),
    ]
    poly = kf.kdb.Polygon()
    poly.assign_hull(points4)
    c.add_polygon(poly, layer="WG")
    assert len(list(c.shapes(get_layer(LAYER.WG)).each())) == 4
    points5 = [
        kf.kdb.DPoint(80, 0),
        kf.kdb.DPoint(90, 0),
        kf.kdb.DPoint(90, 10),
        kf.kdb.DPoint(80, 10),
    ]
    dsimpoly = kf.kdb.DSimplePolygon(points5)
    c.add_polygon(dsimpoly, layer="WG")
    assert len(list(c.shapes(get_layer(LAYER.WG)).each())) == 5

    c._locked = True
    with pytest.raises(LockedError):
        c.add_polygon(points1, layer="WG")


def test_component_all_angle_add_label() -> None:
    c = gf.ComponentAllAngle()
    c.add_label(text="test", position=(10, 20), layer="WG")
    assert len(list(c.shapes(get_layer(LAYER.WG)).each())) == 1

    c.add_label(text="test", position=kf.kdb.DPoint(30, 40), layer="WG")
    assert len(list(c.shapes(get_layer(LAYER.WG)).each())) == 2


def test_component_write_gds() -> None:
    c = gf.Component("test_component_all_angle_write_gds")

    with pytest.warns(UserWarning, match="gdspath and gdsdir have both been specified"):
        c.write_gds(
            gdspath=GDSDIR_TEMP / "test_component_write_gds.gds",
            gdsdir=GDSDIR_TEMP,
        )

    path = c.write_gds(with_metadata=False)
    assert path.exists()

    save_options = kdb.SaveLayoutOptions()
    save_options.write_context_info = False
    path = c.write_gds(save_options=save_options)
    assert path.exists()

    path = c.write_gds(gdsdir=GDSDIR_TEMP)
    assert path.exists()
    assert path.parent == GDSDIR_TEMP

    path = c.write_gds(gdspath=GDSDIR_TEMP / "custom_name.gds")
    assert path.exists()
    assert path.name == "custom_name.gds"


def test_component_copy_child_info() -> None:
    c1 = gf.Component()
    c2 = gf.Component()
    c2.info["test_info"] = "test_value"
    c1.copy_child_info(c2)
    assert c1.info["test_info"] == "test_value"


def test_container() -> None:
    c = gf.Component("test_component")
    c.info["test_info"] = "test_value"

    def test_function(component: gf.Component, **kwargs: Any) -> None:
        component.info["new_info"] = kwargs.get("value", "new_value")

    result = container(c, function=test_function, value="custom_value")

    assert result.info["test_info"] == "test_value"
    assert result.info["new_info"] == "custom_value"
    assert len(result.insts) == 1


def test_component_with_function() -> None:
    c = gf.Component("test_component")
    c.info["test_info"] = "test_value"

    def test_function(component: gf.Component) -> None:
        component.info["new_info"] = "new_value"

    result = component_with_function(c, function=test_function)

    assert result.info["test_info"] == "test_value"
    assert result.info["new_info"] == "new_value"
    assert len(result.insts) == 1


def test_plot() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
    c.plot()


def test_ref_deprecation() -> None:
    c = gf.Component()
    with pytest.warns(DeprecationWarning):
        c.ref(gf.components.straight())


# def test_offset() -> None:
#     c = gf.Component()
#     c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
#     c.offset(layer=(1, 0), distance=10)
#     print(c.dbbox(get_layer((1, 0))))


if __name__ == "__main__":
    test_ref_deprecation()
