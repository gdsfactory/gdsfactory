from __future__ import annotations

from typing import Any

import klayout.db as kdb
import numpy as np
import numpy.typing as npt
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.difftest import difftest
from gdsfactory.generic_tech import LAYER
from gdsfactory.path import Path, _parabolic_transition


def test_path_zero_length() -> None:
    c = gf.components.straight(
        length=0.5e-3, cross_section=gf.cross_section.cross_section
    )
    assert c.area((1, 0)) == 0


def test_path_append() -> None:
    """Append paths."""
    P = gf.Path()
    P.append(gf.path.arc(radius=10, angle=90))  # Circular arc
    P.append(gf.path.straight(length=10))  # Straight section
    P.append(
        gf.path.euler(radius=3, angle=-90, p=1)
    )  # Euler bend (aka "racetrack" curve)
    P.append(gf.path.straight(length=40))
    P.append(gf.path.arc(radius=8, angle=-45))
    P.append(gf.path.straight(length=10))
    P.append(gf.path.arc(radius=8, angle=45))
    P.append(gf.path.straight(length=10))
    assert np.round(P.length(), 3) == 107.697, P.length()


def looploop(num_pts: int = 1000) -> npt.NDArray[np.signedinteger[Any]]:
    """Simple limacon looping curve."""
    t = np.linspace(-np.pi, 0, num_pts)
    r = 20 + 25 * np.sin(t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.array((x, y)).T


@gf.cell
def double_loop() -> Component:
    # Create the path points
    P = gf.Path()
    P.append(gf.path.arc(radius=10, angle=90))
    P.append(gf.path.straight())
    P.append(gf.path.arc(radius=5, angle=-90))
    P.append(looploop(num_pts=1000))
    P.rotate(-45)

    # Create the crosssection
    s0 = gf.Section(width=1.5, offset=0, layer=(2, 0), port_names=("in", "out"))
    s1 = gf.Section(width=0.5, offset=2, layer=(0, 0))
    s2 = gf.Section(width=0.5, offset=4, layer=(1, 0))
    s3 = gf.Section(width=1, offset=0, layer=(3, 0))
    X = gf.CrossSection(sections=(s0, s1, s2, s3))
    return gf.path.extrude(P, X, simplify=0.3)


@gf.cell
def transition() -> Component:
    c = gf.Component()
    s0 = gf.Section(
        width=1.2, offset=0, layer=(2, 0), name="core", port_names=("in1", "out1")
    )
    s1 = gf.Section(width=2.2, offset=0, layer=(3, 0), name="etch")
    s2 = gf.Section(width=1.1, offset=3, layer=(1, 0), name="wg2")
    X1 = gf.CrossSection(sections=(s0, s1, s2))

    # Create the second CrossSection that we want to transition to
    s0 = gf.Section(
        width=1, offset=0, layer=(2, 0), name="core", port_names=("in1", "out1")
    )
    s1 = gf.Section(width=3.5, offset=0, layer=(3, 0), name="etch")
    s2 = gf.Section(width=3, offset=5, layer=(1, 0), name="wg2")
    X2 = gf.CrossSection(sections=(s0, s1, s2))

    Xtrans = gf.path.transition(cross_section1=X1, cross_section2=X2, width_type="sine")
    # Xtrans = gf.cross_section.strip(port_names=('in1', 'out1'))

    P1 = gf.path.straight(length=5)
    P2 = gf.path.straight(length=5)

    wg1 = gf.path.extrude(P1, X1)
    wg2 = gf.path.extrude(P2, X2)

    P4 = gf.path.euler(radius=25, angle=90, p=0.5, use_eff=False)
    wg_trans = gf.path.extrude_transition(P4, Xtrans)

    wg1_ref = c << wg1
    wgt_ref = c << wg_trans
    wgt_ref.connect("in1", wg1_ref.ports["out1"])

    wg2_ref = c << wg2
    wg2_ref.connect("in1", wgt_ref.ports["out1"])
    return c


component_factory = dict(
    transition=transition,
)


component_names = component_factory.keys()


@pytest.fixture(params=component_names)
def component(request: pytest.FixtureRequest) -> Component:
    return component_factory[request.param]()


def test_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    difftest(component)


def test_settings(component: Component, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.to_dict())


def test_layers1() -> None:
    P = gf.path.straight(length=10.001)
    s = gf.Section(width=0.5, offset=0, layer=LAYER.WG, port_names=("in", "out"))
    X = gf.CrossSection(sections=(s,))
    c = gf.path.extrude(P, X, simplify=5e-3)
    assert c.ports["in"].layer == LAYER.WG
    assert c.ports["out"].center[0] == 10.001, c.ports["out"].center[0]


def test_path_add() -> None:
    p1 = gf.path.straight(length=5)
    p2 = gf.path.straight(length=5)
    p3 = p1 + p2
    assert p3.length() == 10

    p2 += p1
    assert p2.length() == 10

    p1 = gf.path.straight(length=5)
    p2 = gf.path.euler(radius=5, angle=45, p=0.5, use_eff=False)

    p = p2 + p1
    assert p.start_angle == 0
    assert p.end_angle == 45


def test_init_with_no_path() -> None:
    path = Path()
    assert np.array_equal(path.points, np.array([(0, 0)], dtype=np.float64))
    assert path.start_angle == 0
    assert path.end_angle == 0


def test_init_with_array() -> None:
    points = [(0, 0), (1, 1), [2, 0]]
    path = Path(points)
    assert np.array_equal(path.points, np.array(points))
    assert path.start_angle == pytest.approx(45.0)
    assert path.end_angle == pytest.approx(-45.0)


def test_init_with_path() -> None:
    original_path = Path([(0, 0), (1, 1), [2, 0]])
    path = Path(original_path)
    assert np.array_equal(path.points, original_path.points)
    assert path.start_angle == original_path.start_angle
    assert path.end_angle == original_path.end_angle


def test_invalid_path() -> None:
    with pytest.raises(ValueError):
        Path("invalid path")


def test_append_path() -> None:
    path1 = Path([(0, 0), (1, 1)])
    path2 = Path([(0, 0), (1, 1)])
    path1.append(path2)
    expected_points = np.array([(0, 0), (1, 1), (2, 2)])
    assert np.array_equal(path1.points, expected_points)


def test_append_points() -> None:
    path = Path([(0, 0), (1, 1)])
    points = [(1, 1), (2, 2)]
    path.append(points)
    expected_points = np.array([(0, 0), (1, 1), (2, 2)])
    assert np.array_equal(path.points, expected_points)


def test_length() -> None:
    path = Path([(0, 0), (1, 1), [2, 0]])
    assert path.length() == pytest.approx(2.8284, rel=1e-3)


def test_dmove() -> None:
    path = Path([(0, 0), (1, 1), [2, 0]])
    path.move((0, 0), (1, 1))
    expected_points = np.array([(1, 1), (2, 2), [3, 1]])
    assert np.array_equal(path.points, expected_points)


def test_drotate() -> None:
    path = Path([(0, 0), (1, 1), [2, 0]])
    path.rotate(90)
    expected_points = np.array([(0, 0), [-1, 1], [0, 2]])
    np.testing.assert_allclose(path.points, expected_points, atol=1e-4)


def test_dmirror() -> None:
    path = Path([(0, 0), (1, 1), [2, 0]])
    path.mirror((0, 0), (0, 1))
    expected_points = np.array([(0, 0), [-1, 1], [-2, 0]])
    np.testing.assert_allclose(path.points, expected_points, atol=1e-4)


def test_path_append_list() -> None:
    p = gf.Path()

    # Create the basic Path components
    left_turn = gf.path.euler(radius=4, angle=90)
    right_turn = gf.path.euler(radius=4, angle=-90)
    p = gf.Path()

    # Create an "S-turn" using a list
    s_turn = [left_turn, right_turn]

    # Repeat the S-turn 3 times by nesting our S-turn list 3x times in another list
    triple_s_turn = [s_turn, s_turn, s_turn]
    p.append(triple_s_turn)
    assert p.length() == 56.545, p.length()


def test_path_init() -> None:
    path_empty = Path()
    assert np.array_equal(path_empty.points, np.array([(0, 0)]))

    path_multiple = Path([(0, 0), (1, 1), (2, 2)])
    expected_points = np.array([(0, 0), (1, 1), (2, 2)])
    assert np.array_equal(path_multiple.points, expected_points)

    with pytest.raises(ValueError):
        Path([(0, 0), (1,)])

    with pytest.raises(ValueError):
        Path([(1,)])

    path_from_path = Path(path_multiple)
    assert np.array_equal(path_from_path.points, expected_points)


def test_path_len() -> None:
    path = Path([(0, 0), (1, 1), (2, 0)])
    assert len(path) == 3

    path = Path()
    assert len(path) == 1


def test_path_bbox_np() -> None:
    path = Path([(0, 0), (1, 1), (2, 0)])
    assert np.array_equal(path.bbox_np(), np.array([(0, 0), (2, 1)]))


def test_path_offset() -> None:
    path = Path([(i, 0) for i in range(10)])
    path.offset(2)
    np.testing.assert_array_almost_equal(
        path.points, np.array([(i, -2) for i in range(10)], dtype=np.float64)
    )

    path = Path([(i, 0) for i in range(10)])
    path.offset(lambda t: t * 9)
    np.testing.assert_array_almost_equal(
        path.points, np.array([(i, -i) for i in range(10)], dtype=np.float64)
    )

    path = Path([(i, 0) for i in range(10)])
    path.offset(0)
    np.testing.assert_array_almost_equal(
        path.points, np.array([(i, 0) for i in range(10)], dtype=np.float64)
    )


def test_rotate() -> None:
    path = Path([(i, 0) for i in range(10)])
    path.rotate(90)
    np.testing.assert_array_almost_equal(
        path.points, np.array([(0, i) for i in range(10)], dtype=np.float64)
    )

    path = Path([(i, 0) for i in range(10)])
    path.rotate(0)
    np.testing.assert_array_almost_equal(
        path.points, np.array([(i, 0) for i in range(10)], dtype=np.float64)
    )

    path = Path([(i, 0) for i in range(10)])
    path.rotate(45)
    expected_points = np.array(
        [
            (
                i * np.cos(np.radians(45)) - 0 * np.sin(np.radians(45)),
                i * np.sin(np.radians(45)) + 0 * np.cos(np.radians(45)),
            )
            for i in range(10)
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_almost_equal(path.points, expected_points)


def test_mirror() -> None:
    path = Path([(i, 0) for i in range(10)])
    path.mirror((0, 0), (0, 1))
    np.testing.assert_array_almost_equal(
        path.points, np.array([(-i, 0) for i in range(10)], dtype=np.float64)
    )

    path = Path([(i, 0) for i in range(10)])
    path.mirror((0, 0), (1, 0))
    np.testing.assert_array_almost_equal(
        path.points, np.array([(i, 0) for i in range(10)], dtype=np.float64)
    )


def test_centerpoint_offset_curve() -> None:
    path = Path([(0, 0), (1, 0), (2, 0)])
    offset_distance = [0.5]
    new_points = path.centerpoint_offset_curve(path.points, offset_distance)
    expected_points = np.array(
        [
            (0, -0.5),
            (1, -0.5),
            (2, -0.5),
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_almost_equal(new_points, expected_points)

    path = Path([(0, 0), (1, 0), (2, 0)])
    offset_distance = np.array([0.5, 1, 0.5], dtype=np.float64)
    new_points = path.centerpoint_offset_curve(path.points, offset_distance)
    expected_points = np.array(
        [
            (0, -0.5),
            (1, -1),
            (2, -0.5),
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_almost_equal(new_points, expected_points)


def test_path_hash() -> None:
    assert hash(Path([(0, 0), (1, 1), (2, 0)])) == hash(Path([(0, 0), (1, 1), (2, 0)]))


def test_path_hash_geometry() -> None:
    assert (
        Path([(0, 0), (1, 1), (2, 0)]).hash_geometry()
        == Path([(0, 0), (1, 1), (2, 0)]).hash_geometry()
    )


def test_path_extrude_transition() -> None:
    path = Path([(0, 0), (1, 0), (1, 1)])
    transition = gf.path.transition(
        cross_section1=gf.cross_section.cross_section,
        cross_section2=gf.cross_section.cross_section,
    )
    c = path.extrude_transition(transition)
    assert c.bbox() == kdb.DBox(0, -0.25, 1.25, 1)


def test_path_copy() -> None:
    path = gf.path.euler()
    path_copy = path.copy()
    assert np.array_equal(path_copy.points, path.points)
    assert path_copy.start_angle == path.start_angle
    assert path_copy.end_angle == path.end_angle


def test_parabolic_transition() -> None:
    y1 = 1.0
    y2 = 3.0
    transition_func = _parabolic_transition(y1, y2)

    t_scalar = 0.5
    assert transition_func(t_scalar) == y1 + np.sqrt(t_scalar) * (y2 - y1)

    t_array = np.array([0.0, 0.5, 1.0])
    expected_array = y1 + np.sqrt(t_array) * (y2 - y1)
    np.testing.assert_array_equal(transition_func(t_array), expected_array)


def test_path_bbox() -> None:
    path = Path([(0, 0), (1, 0), (1, 1)])
    assert path.bbox() == kdb.DBox(0, 0, 1, 1)


def test_path_transform_drans() -> None:
    x = 1.111111111
    path = Path([(0, 0), (x, 0), (x, x)])
    trans = kdb.DTrans(x=1, y=1)
    path.transform(trans)
    np.testing.assert_array_almost_equal(
        path.points, np.array([(1, 1), (x + 1, 1), (x + 1, x + 1)])
    )


def test_path_transform_trans() -> None:
    x = 1.111111111
    path = Path([(0, 0), (x, 0), (x, x)])
    trans = kdb.Trans(x=1000, y=1000)
    path.transform(trans)
    np.testing.assert_array_almost_equal(
        path.points, np.array([(1, 1), (x + 1, 1), (x + 1, x + 1)])
    )


def test_path_transform_icplx() -> None:
    x = 1.111111111
    path = Path([(0, 0), (x, 0), (x, x)])
    trans = kdb.ICplxTrans(x=1000, y=1000)
    path.transform(trans)
    np.testing.assert_array_almost_equal(
        path.points, np.array([(1, 1), (x + 1, 1), (x + 1, x + 1)])
    )
    assert gf.path.Path().kcl is gf.kcl


def test_path_smooth() -> None:
    points = np.array([(-50, 50), (-100, 100), (-100, 200)])

    P = gf.path.smooth(points=points, radius=10, bend=gf.path.euler)
    section = gf.Section(width=20.0, layer=(1, 0))
    X = gf.CrossSection(sections=(section,))

    c = P.extrude(cross_section=X)
    assert np.isclose(c.area((1, 0)), 3404.6317885)


def test_path_angle() -> None:
    p = gf.path.euler(
        radius=5,
        angle=180,
        p=1,
        use_eff=True,
    )
    p.drotate(-90)
    c = p.extrude(cross_section=gf.cross_section.strip)
    assert np.isclose(c.area("WG"), 11.409315999999999)
