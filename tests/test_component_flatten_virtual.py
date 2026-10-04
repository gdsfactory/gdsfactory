import itertools
from pathlib import Path

import klayout.db as kdb
import pytest
from kfactory.exceptions import LockedError

import gdsfactory as gf


def _region(component: gf.Component) -> kdb.Region:
    return kdb.Region(component.kdb_cell.begin_shapes_rec(gf.get_layer((1, 0))))


def _expected(positions: list[tuple[int, int]], component: gf.Component) -> kdb.Region:
    region = kdb.Region()
    dbu = component.kcl.dbu
    for x, y in positions:
        region.insert(
            kdb.Box(
                round(x / dbu),
                round(y / dbu),
                round((x + 1) / dbu),
                round((y + 1) / dbu),
            )
        )
    return region


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("virtual_only", [False, True])
def test_flatten_nested_virtual_instances(
    depth: int, locked: bool, virtual_only: bool, tmp_path: Path
) -> None:
    leaf = gf.Component()
    leaf.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(1, 0))
    child = gf.Component()
    child.add_ref(leaf).dmovex(-10)
    child.create_vinst(leaf).dmovex(10)
    positions = [(-10, 0), (10, 0)]
    children = [child]
    if locked:
        child.lock()
    for level in range(2, depth):
        parent = gf.Component()
        offset = 10 * 3 ** (level - 2)
        parent.add_ref(child).dmovey(offset)
        parent.create_vinst(child).dmovey(-offset)
        positions = [(x, y + sign * offset) for sign in (-1, 1) for x, y in positions]
        child = parent
        children.append(child)
        if locked:
            child.lock()
    root = gf.Component()
    if virtual_only:
        root.create_vinst(child)
    else:
        offset = 10 * 3 ** (depth - 2)
        root.add_ref(child).dmovey(offset)
        root.create_vinst(child).dmovey(-offset)
        positions = [(x, y + sign * offset) for sign in (-1, 1) for x, y in positions]
    snapshots = [
        (
            len(cell.insts),
            len(cell.vinsts),
            cell.locked,
            tuple(v.trans.to_s() for v in cell.vinsts),
        )
        for cell in children
    ]
    expected = _expected(positions, root)
    root.flatten()
    assert (_region(root) ^ expected).is_empty()
    assert len(root.insts) == 0
    assert len(root.vinsts) == 0
    assert snapshots == [
        (
            len(cell.insts),
            len(cell.vinsts),
            cell.locked,
            tuple(v.trans.to_s() for v in cell.vinsts),
        )
        for cell in children
    ]

    filename = root.write_gds(tmp_path / "flattened.gds")
    layout = kdb.Layout()
    layout.read(str(filename))
    top = layout.top_cell()
    assert top is not None
    layer = layout.find_layer(1, 0)
    assert layer is not None
    assert (kdb.Region(top.begin_shapes_rec(layer)) ^ expected).is_empty()


@pytest.mark.parametrize("nested", [False, True])
def test_flatten_unnamed_virtual_merge_false(nested: bool) -> None:
    virtual = gf.ComponentAllAngle()
    virtual.add_polygon([(0, 0), (2, 0), (2, 1), (0, 1)], layer=(1, 0))
    virtual.add_polygon([(1, 0), (3, 0), (3, 1), (1, 1)], layer=(1, 0))
    root = gf.Component()
    if nested:
        child = gf.Component()
        child.create_vinst(virtual)
        child.lock()
        root.add_ref(child)
    else:
        root.create_vinst(virtual)
    root.flatten(merge=False)
    assert _region(root).count() == 2
    assert _region(root).merged().area() * root.kcl.dbu**2 == pytest.approx(3)


def test_flatten_virtual_instances_in_real_array() -> None:
    leaf = gf.Component()
    leaf.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(1, 0))
    child = gf.Component()
    child.create_vinst(leaf).dmovex(2)
    root = gf.Component()
    reference = root.add_ref(child, rows=2, columns=3, row_pitch=10, column_pitch=10)
    reference.dmirror_y()
    root.flatten()
    expected = _expected(
        [
            (2 + 10 * column, -1 - 10 * row)
            for column, row in itertools.product(range(3), range(2))
        ],
        root,
    )
    assert (_region(root) ^ expected).is_empty()


def test_flatten_virtual_real_virtual_chain() -> None:
    leaf = gf.ComponentAllAngle()
    leaf.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(1, 0))
    virtual_middle = gf.ComponentAllAngle()
    (virtual_middle << leaf).dmovex(2)
    real_middle = gf.Component()
    real_middle.create_vinst(virtual_middle).dmovex(3)
    virtual_outer = gf.ComponentAllAngle()
    (virtual_outer << real_middle).dmovex(4)
    root = gf.Component()
    root.create_vinst(virtual_outer).dmovex(5)
    root.flatten()
    assert (_region(root) ^ _expected([(14, 0)], root)).is_empty()


@pytest.mark.parametrize("angle", [0, 37, 90])
def test_flatten_rotated_nested_virtual_instance(angle: int) -> None:
    leaf = gf.Component()
    leaf.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(1, 0))
    child = gf.Component()
    child.create_vinst(leaf).dmovex(2)
    root = gf.Component()
    reference = root.add_ref(child)
    reference.drotate(angle)
    transform = reference.dcplx_trans
    expected = _expected([(2, 0)], root)
    expected.transform(kdb.ICplxTrans(transform, root.kcl.dbu))
    root.flatten()
    assert (_region(root) ^ expected).is_empty()


def test_flatten_locked_component_preserves_virtual_instances() -> None:
    leaf = gf.Component()
    leaf.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(1, 0))
    root = gf.Component()
    root.create_vinst(leaf)
    root.lock()
    with pytest.raises(LockedError):
        root.flatten()
    assert len(root.vinsts) == 1
    assert len(root.insts) == 0
