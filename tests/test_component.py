from typing import Any

import kfactory as kf
import klayout.db as kdb
import numpy as np
import pytest
from kfactory.exceptions import LockedError

import gdsfactory as gf
from gdsfactory.component import (
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
    c1 = gf.components.mzi(delta_length=10)
    c2 = c1.copy()
    assert c1.settings["delta_length"] == 10
    assert c1.info == c2.info

    assert len(c1.ports) == len(c2.ports)
    assert len(list(c1.insts)) == len(list(c2.insts)), (
        f"{len(list(c1.insts))} != {len(list(c2.insts))}"
    )


def test_component_all_angle_copy() -> None:
    c1 = gf.components.straight_all_angle(length=10)
    c2 = c1.dup()
    assert c1.settings["length"] == 10
    assert c1.info == c2.info

    assert len(c1.ports) == len(c2.ports)
    assert len(list(c1.insts)) == len(list(c2.insts)), (
        f"{len(list(c1.insts))} != {len(list(c2.insts))}"
    )


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

    c2 = gf.c.rectangle(size=(9, 9), centered=True, layer=layer).copy()
    c2.flatten()
    c2_area = c2.area(layer=layer)

    c2.trim(left=-5, right=5, top=5, bottom=-5, flatten=True)
    assert c2_area == c2.area(layer=layer), f"{c2_area} != {c2.area(layer=layer)}"


def test_remove_layers() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(2, 0))

    c.remove_layers(layers=[(2, 0)])
    assert c.area((1, 0)) == 100, f"{c.area((1, 0))}"
    assert c.area((2, 0)) == 0, f"{c.area((2, 0))}"


def test_locked_cell() -> None:
    c = gf.Component()
    c.locked = True

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

    with pytest.raises(LockedError):
        c.get_polygons_points(merge=True)

    with pytest.raises(LockedError):
        c.get_polygons(merge=True)

    with pytest.raises(LockedError):
        c.add_ref(gf.Component())

    with pytest.raises(LockedError):
        c.trim(left=-5, right=5, top=5, bottom=-5)

    with pytest.raises(LockedError):
        c.absorb(c.add_ref(gf.Component()))

    with pytest.raises(LockedError):
        c.absorb(c.add_ref(gf.Component()))

    with pytest.raises(LockedError):
        c.add(kf.DInstance(gf.Component().kcl, kdb.Instance()))


def test_locked_cell_all_angle() -> None:
    c = gf.ComponentAllAngle()
    c.locked = True

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

    assert ref.center == ref.center
    assert ref.x == ref.dx
    assert ref.y == ref.dy
    assert ref.xmin == ref.dxmin
    assert ref.xmax == ref.dxmax
    assert ref.ymin == ref.dymin
    assert ref.ymax == ref.dymax
    assert ref.xsize == ref.dxsize
    assert ref.ysize == ref.dysize
    assert ref.size_info._bf() == ref.dsize_info._bf()


def test_component_reference_setters() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight

    ref.x = 5.0
    assert ref.dx == 5.0
    ref.y = 10.0
    assert ref.dy == 10.0


def test_component_reference_transformations() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight

    original_pos = ref.center
    ref.move((1.0, 2.0))
    assert ref.center != original_pos

    ref.rotate(90)
    ref.mirror()


def test_component_reference_name() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight

    ref.name = "test_instance"
    assert ref.name == "test_instance"


def test_component_reference_flatten() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight
    ref.flatten()


def test_component_references_getitem() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight
    ref2 = c << straight
    ref2.name = "test_ref"

    assert c.insts[0].instance == ref.instance
    assert c.insts[1].instance == ref2.instance

    assert c.insts["test_ref"].instance == ref2.instance


def test_component_references_iter() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=10).copy()
    ref = c << straight
    ref2 = c << straight

    refs = list(c.insts)
    assert len(refs) == 2
    assert refs[0].instance == ref.instance
    assert refs[1].instance == ref2.instance
    assert all(isinstance(r, kf.DInstance) for r in refs)


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
    assert port1.width == 0.5
    assert port1.orientation == 90
    assert port1.port_type == "optical"
    assert port1.center == (10, 20)

    port2 = c.add_port(
        name="p2",
        center=kdb.DPoint(30, 40),
        width=1.0,
        orientation=0,
        layer="WG",
        port_type="electrical",
    )
    assert port2.name == "p2"
    assert port2.width == 1.0
    assert port2.orientation == 0
    assert port2.port_type == "electrical"
    assert port2.center == (30, 40)

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
    assert port3.center == (50, 60)

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

    c.locked = True
    with pytest.raises(LockedError):
        c.add_polygon(points1, layer="WG")


def test_component_all_angle_add_label() -> None:
    c = gf.ComponentAllAngle()
    c.add_label(text="test", position=(10, 20), layer="WG")
    assert len(list(c.shapes(get_layer(LAYER.WG)).each())) == 1

    c.add_label(text="test", position=kf.kdb.DPoint(30, 40), layer="WG")
    assert len(list(c.shapes(get_layer(LAYER.WG)).each())) == 2


def test_component_write_gds() -> None:
    c = gf.Component(name="test_component_all_angle_write_gds")

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
    c2.info["test_info2"] = "test_value2"
    c1.copy_child_info(c2)
    assert c1.info["test_info"] == "test_value"
    assert c1.info["test_info2"] == "test_value2"


def test_container() -> None:
    c = gf.Component()
    c.info["test_info"] = "test_value"

    def test_function(component: gf.Component, **kwargs: Any) -> None:
        component.info["new_info"] = kwargs.get("value", "new_value")

    result = container(c, function=test_function, value="custom_value")

    assert result.info["test_info"] == "test_value"
    assert result.info["new_info"] == "custom_value"
    assert len(result.insts) == 1


def test_plot() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
    c.plot()
    c.to_graphviz()
    c.to_dict(True)


def test_offset() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
    c.offset(layer=(1, 0), distance=10)

    c.kcl.layout.end_changes()

    assert c.dbbox(get_layer((1, 0))) == kdb.Box(-10, -10, 20, 20)


def test_over_under() -> None:
    c = gf.Component()

    c.add_polygon([(0, 0), (0, 5), (5, 5), (5, 0)], layer=(1, 0))
    c.add_polygon([(5.001, 0), (5.001, 5), (10, 5), (10, 0)], layer=(1, 0))

    c.add_polygon([(0, 0), (0, 0.0005), (0, 0.0005), (0.0005, 0)], layer=(1, 0))

    c.over_under(layer=(1, 0), distance=1)

    assert len(c.shapes(get_layer((1, 0)))) == 1
    assert c.dbbox(get_layer((1, 0))) == kdb.Box(0, 0, 10, 5)


def test_component_all_angle_plot() -> None:
    c = gf.ComponentAllAngle()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
    c.plot()


def test_component_to_3d() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
    c.to_3d()


def test_remap_layers() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(2, 0))

    c.remap_layers({(1, 0): (3, 0)})
    assert c.area((1, 0)) == 0, f"{c.area((1, 0))}"
    assert c.area((2, 0)) == 100, f"{c.area((2, 0))}"
    assert c.area((3, 0)) == 100, f"{c.area((3, 0))}"


def test_copy_layers() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))

    c2 = gf.Component()
    c2.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(5, 0))
    ref = c.add_ref(c2)
    c.copy_layers({(1, 0): (3, 0), (5, 0): (6, 0)}, recursive=False)
    assert c.area((1, 0)) == 100, f"{c.area((1, 0))}"
    assert c.area((3, 0)) == 100, f"{c.area((3, 0))}"
    component_ref_cell = gf.Component(base=ref.cell.base)
    assert component_ref_cell.area((5, 0)) == 100, f"{component_ref_cell.area((5, 0))}"
    assert component_ref_cell.area((6, 0)) == 0, f"{component_ref_cell.area((6, 0))}"

    c3 = gf.Component()
    ref2 = c3.add_ref(c)
    c3.copy_layers({(1, 0): (3, 0)}, recursive=True)
    assert c3.area((1, 0)) == 100, f"{c3.area((1, 0))}"
    assert c3.area((3, 0)) == 100, f"{c3.area((3, 0))}"
    component_ref_cell = gf.Component(base=ref2.cell.base)
    assert component_ref_cell.area((1, 0)) == 100, f"{component_ref_cell.area((1, 0))}"
    assert component_ref_cell.area((3, 0)) == 100, f"{component_ref_cell.area((3, 0))}"


def test_get_labels() -> None:
    c = gf.Component()
    c.add_label(text="test1", position=(10, 20), layer="WG")
    c.add_label(text="test2", position=(30, 40), layer="WG")

    labels = c.get_labels(layer="WG", recursive=False)
    assert len(labels) == 2
    assert labels[0].string == "test1"
    assert labels[0].x == 10
    assert labels[0].y == 20
    assert labels[1].string == "test2"
    assert labels[1].x == 30
    assert labels[1].y == 40

    c2 = gf.Component()
    ref = c2 << c
    ref.move((100, 100))

    labels = c2.get_labels(layer="WG", recursive=True)
    assert len(labels) == 2
    assert labels[0].string == "test1"
    assert labels[0].x == 110
    assert labels[0].y == 120
    assert labels[1].string == "test2"
    assert labels[1].x == 130
    assert labels[1].y == 140


def test_get_boxes() -> None:
    c = gf.Component()
    box = kf.kdb.DBox(0, 0, 10, 10)
    c.shapes(get_layer((1, 0))).insert(box)

    boxes = c.get_boxes(layer=(1, 0), recursive=False)
    assert len(boxes) == 1
    assert boxes[0].left == 0
    assert boxes[0].right == 10
    assert boxes[0].top == 10
    assert boxes[0].bottom == 0

    c2 = gf.Component()
    ref = c2 << c
    ref.move((100, 100))

    boxes = c2.get_boxes(layer=(1, 0), recursive=True)
    assert len(boxes) == 1
    assert boxes[0].left == 100
    assert boxes[0].right == 110
    assert boxes[0].top == 110
    assert boxes[0].bottom == 100


def test_get_paths() -> None:
    c = gf.Component()
    path = kf.kdb.DPath([kf.kdb.DPoint(0, 0), kf.kdb.DPoint(10, 10)], 1)
    c.shapes(get_layer((1, 0))).insert(path)

    paths = c.get_paths(layer=(1, 0), recursive=False)
    assert len(paths) == 1
    points = list(paths[0].each_point())
    assert points[0].x == 0
    assert points[0].y == 0
    assert points[1].x == 10
    assert points[1].y == 10

    c2 = gf.Component()
    ref = c2 << c
    ref.move((100, 100))

    paths = c2.get_paths(layer=(1, 0), recursive=True)
    assert len(paths) == 1
    points = list(paths[0].each_point())
    assert points[0].x == 100
    assert points[0].y == 100
    assert points[1].x == 110
    assert points[1].y == 110


def test_component_all_angle_flatten() -> None:
    c = gf.ComponentAllAngle()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))

    c2 = gf.ComponentAllAngle()
    c2.add_polygon([(0, 0), (0, 5), (5, 5), (5, 0)], layer=(2, 0))

    c3 = gf.ComponentAllAngle()
    c3 << c
    c3 << c2

    c3.flatten()

    assert len(list(c.shapes(get_layer((1, 0))))) == 1
    assert len(list(c2.shapes(get_layer((2, 0))))) == 1

    assert len(list(c3.shapes(get_layer((1, 0))))) == 1
    assert len(list(c3.shapes(get_layer((2, 0))))) == 1

    poly1 = next(iter(c3.shapes(get_layer((1, 0)))))
    poly2 = next(iter(c3.shapes(get_layer((2, 0)))))

    assert poly1.bbox().left == 0
    assert poly1.bbox().right == 10
    assert poly2.bbox().left == 0
    assert poly2.bbox().right == 5


def test_component_all_angle_add_polygon_get_polygon() -> None:
    c = gf.ComponentAllAngle()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
    c.add_polygon([(5, 5), (5, 15), (15, 15), (15, 5)], layer=(2, 0))

    polygons = c.get_polygons(layer=(1, 0))
    assert len(polygons) == 1
    assert polygons[0].bbox() == kdb.DBox(0, 0, 10, 10)

    polygons = c.get_polygons(layer=(2, 0))
    assert len(polygons) == 1
    assert polygons[0].bbox() == kdb.DBox(5, 5, 15, 15)


def test_component_add_ref_raises() -> None:
    c = gf.Component()
    c2 = gf.Component()

    with pytest.raises(ValueError):
        c.add_ref(c2, rows=2, row_pitch=0)

    with pytest.raises(ValueError):
        c.add_ref(c2, columns=2, column_pitch=0)


def test_component_absorb() -> None:
    c = gf.Component()
    c2 = gf.Component()
    c2.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))

    ref = c.add_ref(c2)

    c.absorb(ref)
    assert len(list(c.insts)) == 0
    assert len(list(c.shapes(get_layer((1, 0))))) == 1


def test_layers() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer=(1, 0))
    assert c.layers == [(1, 0)]


def test_get_netlist_recursive() -> None:
    c = gf.Component()
    child = gf.Component()
    c << child
    assert c.name in c.get_netlist(recursive=True)


if __name__ == "__main__":
    pytest.main([__file__])
