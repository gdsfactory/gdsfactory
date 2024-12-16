from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.difftest import difftest
from gdsfactory.generic_tech import LAYER
from gdsfactory.path import Path


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
    P.drotate(-45)

    # Create the crosssection
    s0 = gf.Section(width=1.5, offset=0, layer=(2, 0), ports=("in", "out"))
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


@pytest.fixture(params=component_names, scope="function")
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
    assert c.ports["out"].dcenter[0] == 10.001, c.ports["out"].dcenter[0]


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
    assert np.array_equal(path.points, np.array([[0, 0]], dtype=np.float64))
    assert path.start_angle == 0
    assert path.end_angle == 0


def test_init_with_array() -> None:
    points = [[0, 0], [1, 1], [2, 0]]
    path = Path(points)
    assert np.array_equal(path.points, np.array(points))
    assert path.start_angle == pytest.approx(45.0)
    assert path.end_angle == pytest.approx(-45.0)


def test_init_with_path() -> None:
    original_path = Path([[0, 0], [1, 1], [2, 0]])
    path = Path(original_path)
    assert np.array_equal(path.points, original_path.points)
    assert path.start_angle == original_path.start_angle
    assert path.end_angle == original_path.end_angle


def test_invalid_path() -> None:
    with pytest.raises(ValueError):
        Path("invalid path")


def test_append_path() -> None:
    path1 = Path([[0, 0], [1, 1]])
    path2 = Path([[0, 0], [1, 1]])
    path1.append(path2)
    expected_points = np.array([[0, 0], [1, 1], [2, 2]])
    assert np.array_equal(path1.points, expected_points)


def test_append_points() -> None:
    path = Path([[0, 0], [1, 1]])
    points = [[1, 1], [2, 2]]
    path.append(points)
    expected_points = np.array([[0, 0], [1, 1], [2, 2]])
    assert np.array_equal(path.points, expected_points)


def test_length() -> None:
    path = Path([[0, 0], [1, 1], [2, 0]])
    assert path.length() == pytest.approx(2.8284, rel=1e-3)


def test_dmove() -> None:
    path = Path([[0, 0], [1, 1], [2, 0]])
    path.move((0, 0), (1, 1))
    expected_points = np.array([[1, 1], [2, 2], [3, 1]])
    assert np.array_equal(path.points, expected_points)


def test_drotate() -> None:
    path = Path([[0, 0], [1, 1], [2, 0]])
    path.rotate(90)
    expected_points = np.array([[0, 0], [-1, 1], [0, 2]])
    np.testing.assert_allclose(path.points, expected_points, atol=1e-4)


def test_dmirror() -> None:
    path = Path([[0, 0], [1, 1], [2, 0]])
    path.mirror((0, 0), (0, 1))
    expected_points = np.array([[0, 0], [-1, 1], [-2, 0]])
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


if __name__ == "__main__":
    test_path_append_list()
