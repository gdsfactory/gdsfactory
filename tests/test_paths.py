from __future__ import annotations

import jsondiff
import numpy as np
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.difftest import difftest


def test_append() -> None:
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


def looploop(num_pts=1000):
    """Simple limacon looping curve."""
    t = np.linspace(-np.pi, 0, num_pts)
    r = 20 + 25 * np.sin(t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.array((x, y)).T


@cell
def double_loop() -> Component:
    # Create the path points
    P = gf.Path()
    P.append(gf.path.arc(radius=10, angle=90))
    P.append(gf.path.straight())
    P.append(gf.path.arc(radius=5, angle=-90))
    P.append(looploop(num_pts=1000))
    P.rotate(-45)

    # Create the crosssection
    s1 = gf.Section(width=0.5, offset=2, layer=(0, 0))
    s2 = gf.Section(width=0.5, offset=4, layer=(1, 0))
    s3 = gf.Section(width=1, offset=0, layer=(3, 0))
    X = gf.CrossSection(
        width=1.5,
        offset=0,
        layer=(2, 0),
        port_names=["in", "out"],
        sections=[s1, s2, s3],
    )

    return gf.path.extrude(P, X, simplify=0.3)


@cell
def transition() -> Component:
    c = gf.Component()
    s1 = gf.Section(width=2.2, offset=0, layer=(3, 0), name="etch")
    s2 = gf.Section(width=1.1, offset=3, layer=(1, 0), name="wg2")
    X1 = gf.CrossSection(
        width=1.2,
        offset=0,
        layer=(2, 0),
        name="wg",
        port_names=("in1", "out1"),
        sections=[s1, s2],
    )

    # Create the second CrossSection that we want to transition to
    s1 = gf.Section(width=3.5, offset=0, layer=(3, 0), name="etch")
    s2 = gf.Section(width=3, offset=5, layer=(1, 0), name="wg2")
    X2 = gf.CrossSection(
        width=1,
        offset=0,
        layer=(2, 0),
        name="wg",
        port_names=("in1", "out1"),
        sections=[s1, s2],
    )

    Xtrans = gf.path.transition(cross_section1=X1, cross_section2=X2, width_type="sine")
    # Xtrans = gf.cross_section.strip(port_names=('in1', 'out1'))

    P1 = gf.path.straight(length=5)
    P2 = gf.path.straight(length=5)

    wg1 = gf.path.extrude(P1, X1)
    wg2 = gf.path.extrude(P2, X2)

    P4 = gf.path.euler(radius=25, angle=45, p=0.5, use_eff=False)
    wg_trans = gf.path.extrude(P4, Xtrans)

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
def component(request) -> Component:
    return component_factory[request.param]()


def test_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    difftest(component)


def test_settings(component: Component, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.to_dict())


def test_layers1() -> None:
    P = gf.path.straight(length=10.001)
    X = gf.CrossSection(width=0.5, offset=0, layer=(3, 0), port_names=("in", "out"))
    c = gf.path.extrude(P, X, simplify=5e-3)
    assert c.ports["in"].layer == (3, 0)
    assert c.ports["out"].center[0] == 10.001, c.ports["out"].center[0]


def test_layers2() -> None:
    P = gf.path.straight(length=10.001)
    X = gf.cross_section.strip(snap_to_grid=5e-3)
    c = gf.path.extrude(P, X, simplify=5e-3)
    assert c.ports["o1"].layer == (1, 0)
    assert c.ports["o2"].center[0] == 10.0, c.ports["o2"].center[0]


def test_copy() -> None:
    x1 = gf.CrossSection(width=0.5, offset=0, layer=(3, 0), port_names=("in", "out"))
    x2 = x1.copy()

    d = jsondiff.diff(x1.model_dump(), x2.model_dump())
    assert len(d) == 0, d


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


if __name__ == "__main__":
    # test_append()
    # c = transition()

    p1 = gf.path.straight(length=5)
    p2 = gf.path.euler(radius=5, angle=45, p=0.5, use_eff=False)
    p = p2 + p1
    # assert p.start_angle == 45
    # assert p.end_angle == 0
    c = p.extrude(cross_section="strip")
    c.show(show_ports=False)
