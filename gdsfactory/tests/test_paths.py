import numpy as np
import pytest
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.difftest import difftest


@cell
def test_path():
    P = gf.Path()
    P.append(gf.path.arc(radius=10, angle=90))  # Circular arc
    P.append(gf.path.straight(length=10))  # Straight section
    P.append(gf.path.euler(radius=3, angle=-90))  # Euler bend (aka "racetrack" curve)
    P.append(gf.path.straight(length=40))
    P.append(gf.path.arc(radius=8, angle=-45))
    P.append(gf.path.straight(length=10))
    P.append(gf.path.arc(radius=8, angle=45))
    P.append(gf.path.straight(length=10))
    P.length()
    P.length()
    assert np.isclose(P.length(), 107.69901058617913)

    # Create a blank CrossSection
    X = gf.CrossSection()

    # Add a single "section" to the cross-section
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


@cell
def no_rename():
    p = gf.path.arc()

    # Create a blank CrossSection
    X = gf.CrossSection()

    # Add a a few "sections" to the cross-section
    X.add(width=1, offset=0, layer=0, ports=("in", "out"))
    X.add(width=3, offset=2, layer=2)
    X.add(width=3, offset=-2, layer=2)

    # Combine the Path and the CrossSection
    straight = gf.path.extrude(p, cross_section=X, rename_ports=False)
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
    X.add(width=0.5, offset=2, layer=0, ports=[None, None])
    X.add(width=0.5, offset=4, layer=1, ports=[None, "out2"])
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
    # WG_trans = P4.extrude(Xtrans)

    wg1_ref = c << wg1
    wg2_ref = c << wg2
    wgt_ref = c << wg_trans

    wgt_ref.connect("W0", wg1_ref.ports["E0"])
    wg2_ref.connect("W0", wgt_ref.ports["E0"])
    return c


component_factory = gf.types.get_name_to_function_dict(
    test_path, rename, no_rename, double_loop, transition
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
    data_regression.check(component.get_settings())


def test_ports(component: Component, num_regression: NumericRegressionFixture) -> None:
    """Avoid regressions in port names and locations."""
    if component.ports:
        num_regression.check(component.get_ports_array())


def test_layers():
    P = gf.path.straight(length=10.001)
    X = gf.CrossSection()
    X.add(width=0.5, offset=0, layer=gf.LAYER.SLAB90, ports=["in", "out"])
    c = gf.path.extrude(P, X, simplify=5e-3, snap_to_grid_nm=5)
    assert c.ports["W0"].layer == gf.LAYER.SLAB90
    assert c.ports["E0"].position[0] == 10.0
    return c


if __name__ == "__main__":
    c = test_layers()
    c.show()
    # c = transition()
    # c = double_loop()
    # c = rename()
    # c.pprint()
    # c.show()
