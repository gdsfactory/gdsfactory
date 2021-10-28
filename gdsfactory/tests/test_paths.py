import numpy as np
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.difftest import difftest


@cell
def test_path():
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
    P.length()
    P.length()
    assert np.isclose(P.length(), 107.69901058617913), P.length()

    X = gf.CrossSection()
    X.add(width=1, offset=0, layer=0)

    c = gf.path.extrude(P, X)
    return c


@cell
def rename():
    p = gf.path.arc()

    # Create a blank CrossSection
    X = gf.CrossSection()

    # Add a a few "sections" to the cross-section
    X.add(width=1, offset=0, layer=0, ports=("in", "out"))
    X.add(width=3, offset=2, layer=2)
    X.add(width=3, offset=-2, layer=2)

    # Combine the Path and the CrossSection
    straight = gf.path.extrude(p, cross_section=X)
    return straight


def looploop(num_pts=1000):
    """Simple limacon looping curve"""
    t = np.linspace(-np.pi, 0, num_pts)
    r = 20 + 25 * np.sin(t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    points = np.array((x, y)).T
    return points


@cell
def double_loop():
    # Create the path points
    P = gf.Path()
    P.append(gf.path.arc(radius=10, angle=90))
    P.append(gf.path.straight())
    P.append(gf.path.arc(radius=5, angle=-90))
    P.append(looploop(num_pts=1000))
    P.rotate(-45)

    # Create the crosssection
    X = gf.CrossSection()
    X.add(width=0.5, offset=2, layer=0)
    X.add(width=0.5, offset=4, layer=1)
    X.add(width=1.5, offset=0, layer=2, ports=["in", "out"])
    X.add(width=1, offset=0, layer=3)

    c = gf.path.extrude(P, X, simplify=0.3)
    return c


@cell
def transition():
    c = gf.Component()
    X1 = gf.CrossSection()
    X1.add(width=1.2, offset=0, layer=2, name="wg", ports=("in1", "out1"))
    X1.add(width=2.2, offset=0, layer=3, name="etch")
    X1.add(width=1.1, offset=3, layer=1, name="wg2")

    # Create the second CrossSection that we want to transition to
    X2 = gf.CrossSection()
    X2.add(width=1, offset=0, layer=2, name="wg", ports=("in2", "out2"))
    X2.add(width=3.5, offset=0, layer=3, name="etch")
    X2.add(width=3, offset=5, layer=1, name="wg2")

    Xtrans = gf.path.transition(cross_section1=X1, cross_section2=X2, width_type="sine")

    P1 = gf.path.straight(length=5)
    P2 = gf.path.straight(length=5)

    wg1 = gf.path.extrude(P1, X1)
    wg2 = gf.path.extrude(P2, X2)

    P4 = gf.path.euler(radius=25, angle=45, p=0.5, use_eff=False)
    wg_trans = gf.path.extrude(P4, Xtrans)

    # print("wg1", wg1)
    # print("wg2", wg2)
    # print("wg3", wg_trans)
    # wg_trans.pprint()

    wg1_ref = c << wg1
    wg2_ref = c << wg2
    wgt_ref = c << wg_trans

    wgt_ref.connect("in2", wg1_ref.ports["out1"])
    wg2_ref.connect("in2", wgt_ref.ports["out1"])
    return c


component_factory = dict(
    test_path=test_path,
    rename=rename,
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


def test_layers1():
    P = gf.path.straight(length=10.001)
    X = gf.CrossSection()
    X.add(width=0.5, offset=0, layer=gf.LAYER.SLAB90, ports=["in", "out"])
    c = gf.path.extrude(P, X, simplify=5e-3)
    assert c.ports["in"].layer == gf.LAYER.SLAB90
    assert c.ports["out"].position[0] == 10.001, c.ports["out"].position[0]
    return c


def test_layers2():
    P = gf.path.straight(length=10.001)
    X = gf.cross_section.strip(snap_to_grid=5e-3)
    c = gf.path.extrude(P, X, simplify=5e-3)
    assert c.ports["o1"].layer == (1, 0)
    assert c.ports["o2"].position[0] == 10.0, c.ports["o2"].position[0]
    return c


def test_copy():
    X = gf.CrossSection()
    X.add(width=0.5, offset=0, layer=gf.LAYER.SLAB90, ports=["in", "out"])
    x2 = X.copy()
    assert x2


if __name__ == "__main__":
    c = transition()
    # c = test_path()
    # print(c.name)

    # test_copy()
    # c = test_layers2()
    # c = transition()
    # c = double_loop()
    # c = rename()
    # c.pprint()
    c.show()
