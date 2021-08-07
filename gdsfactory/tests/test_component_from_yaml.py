import itertools as it

import jsondiff
import numpy as np
import pytest
from omegaconf import OmegaConf
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

from gdsfactory.component import Component
from gdsfactory.component_from_yaml import component_from_yaml, sample_mmis
from gdsfactory.difftest import difftest

sample_connections = """
name: sample_connections

instances:
    wgw:
      component: straight
      settings:
        length: 1
    wgn:
      component: straight
      settings:
        length: 0.5

connections:
    wgw,E0: wgn,W0

"""

#
#        __Lx__
#       |      |
#       Ly     Lyr
#       |      |
#  CP1==|      |==CP2
#       |      |
#       Ly     Lyr
#       |      |
#      DL/2   DL/2
#       |      |
#       |__Lx__|
#


sample_mirror_simple = """
name: sample_mirror_simple

instances:
    w:
        component: straight

    b:
        component: bend_circular

placements:
    b:
        mirror: True
        port: W0

connections:
    b,W0: w,E0

"""


def test_sample() -> Component:
    c = component_from_yaml(sample_mmis)
    assert len(c.get_dependencies()) == 6
    assert len(c.ports) == 2
    return c


def test_connections() -> Component:
    c = component_from_yaml(sample_connections)
    assert len(c.get_dependencies()) == 2
    assert len(c.ports) == 0
    return c


sample_2x2_connections = """
name: connections_2x2_solution

instances:
    mmi_bottom:
      component: mmi2x2
      settings:
            length_mmi: 5
    mmi_top:
      component: mmi2x2
      settings:
            length_mmi: 10

placements:
    mmi_top:
        x: 100
        y: 100

routes:
    optical:
        links:
            mmi_bottom,E0: mmi_top,W0
            mmi_bottom,E1: mmi_top,W1

        settings:
            waveguide: strip_heater

"""


def test_connections_2x2() -> Component:
    c = component_from_yaml(sample_2x2_connections)
    assert len(c.get_dependencies()) == 8
    assert len(c.ports) == 0

    length = c.routes["mmi_bottom,E1:mmi_top,W1"]
    assert np.isclose(length, 166.098), f"{length}"
    return c


sample_different_factory = """
name: sample_different_factory

instances:
    bl:
      component: pad
    tl:
      component: pad
    br:
      component: pad
    tr:
      component: pad

placements:
    tl:
        x: 0
        y: 200

    br:
        x: 400
        y: 400

    tr:
        x: 400
        y: 600

routes:
    electrical:
        settings:
            separation: 10
            waveguide: metal_routing
        links:
            tl,E: tr,W
            bl,E: br,W
    optical:
        settings:
            radius: 100
            waveguide: strip
        links:
            bl,S: br,E

"""


def test_connections_different_factory() -> Component:
    c = component_from_yaml(sample_different_factory)
    lengths = [696.8, 696.8, 1204.013]
    # print(c.routes["tl,E:tr,W"])
    # print(c.routes["bl,E:br,W"])
    # print(c.routes["bl,S:br,E"])

    assert np.isclose(c.routes["tl,E:tr,W"], lengths[0])
    assert np.isclose(c.routes["bl,E:br,W"], lengths[1])
    assert np.isclose(c.routes["bl,S:br,E"], lengths[2])

    return c


sample_different_link_factory = """
name: sample_path_length_matching

instances:
    bl:
      component: pad
    tl:
      component: pad
    br:
      component: pad
    tr:
      component: pad

placements:
    tl:
        x: 0
        y: 200

    br:
        x: 900
        y: 400

    tr:
        x: 900
        y: 600

routes:
    route1:
        routing_strategy: get_bundle_path_length_match
        settings:
            radius: 10
            extra_length: 500
        links:
            tl,E: tr,W
            bl,E: br,W

"""


def test_connections_different_link_factory() -> Component:
    c = component_from_yaml(sample_different_link_factory)

    length = 1720.794
    assert np.isclose(c.routes["tl,E:tr,W"], length), f"{c.routes['tl,E:tr,W']}"
    assert np.isclose(c.routes["bl,E:br,W"], length)
    return c


sample_waypoints = """
name: sample_waypoints

instances:
    t:
      component: pad_array
      settings:
          port_list:
            - S
    b:
      component: pad_array
      settings:
          port_list:
            - N

placements:
    t:
        x: -250
        y: 1000
routes:
    route1:
        routing_strategy: get_bundle_from_waypoints
        settings:
            waypoints:
                - [0, 300]
                - [400, 300]
                - [400, 400]
                - [-250, 400]
            auto_widen: False
        links:
            b,N0: t,S0
            b,N1: t,S1
"""


sample_docstring = """
name: sample_docstring

instances:
    mmi_bot:
      component: mmi1x2
      settings:
        width_mmi: 5
        length_mmi: 11
    mmi_top:
      component: mmi1x2
      settings:
        width_mmi: 6
        length_mmi: 22

placements:
    mmi_top:
        port: W0
        x: 0
        y: 0
    mmi_bot:
        port: W0
        x: mmi_top,E1
        y: mmi_top,E1
        dx: 40
        dy: -40
routes:
    optical:
        links:
            mmi_top,E0: mmi_bot,W0
"""


sample_regex_connections = """
name: sample_regex_connections

instances:
    left:
      component: nxn
      settings:
        west: 0
        east: 3
        ysize: 20
    right:
      component: nxn
      settings:
        west: 3
        east: 0
        ysize: 20

placements:
    right:
        x: 20
routes:
    optical:
        links:
            left,E:0:2: right,W:0:2
"""

sample_regex_connections_backwards = """
name: sample_regex_connections_backwards

instances:
    left:
      component: nxn
      settings:
        west: 0
        east: 3
        ysize: 20
    right:
      component: nxn
      settings:
        west: 3
        east: 0
        ysize: 20

placements:
    right:
        x: 20
routes:
    optical:
        links:
            left,E:2:0: right,W:2:0
"""


def test_connections_regex() -> Component:
    c = component_from_yaml(sample_regex_connections)
    route_names = ["left,E0:right,W0", "left,E1:right,W1", "left,E2:right,W2"]

    length = 12.0
    for route_name in route_names:
        assert np.isclose(c.routes[route_name], length)
    return c


def test_connections_regex_backwargs() -> Component:
    c = component_from_yaml(sample_regex_connections_backwards)
    route_names = ["left,E0:right,W0", "left,E1:right,W1", "left,E2:right,W2"]

    length = 12.0
    for route_name in route_names:
        assert np.isclose(c.routes[route_name], length), c.routes[route_name]
    return c


def test_connections_waypoints() -> Component:
    c = component_from_yaml(sample_waypoints)

    length = 1937.196
    route_name = "b,N0:t,S0"
    assert np.isclose(c.routes[route_name], length), c.routes[route_name]
    return c


def test_docstring_sample() -> Component:
    c = component_from_yaml(sample_docstring)
    route_name = "mmi_top,E0:mmi_bot,W0"
    length = 72.348
    assert np.isclose(c.routes[route_name], length), c.routes[route_name]
    return c


yaml_fail = """
name: yaml_fail
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 10
    mmi_short:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5

placements:
    mmi_short:
        port: W0
        x: mmi_long,E1
        y: mmi_long,E1
    mmi_long:
        port: W0
        x: mmi_short,E1
        y: mmi_short,E1
        dx : 10
        dy: 20
"""
#                      ______
#                     |      |
#           dx  W0----| short|
#                |    |______|
#                | dy
#   ______ north |
#   |     |
#   |long |
#   |_____|
#       east

yaml_anchor = """
name: yaml_anchor
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 10
    mmi_short:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5

placements:
    mmi_short:
        port: E0
        x: 0
        y: 0
    mmi_long:
        port: W0
        x: mmi_short,east
        y: mmi_short,north
        dx : 10
        dy: 10
"""

# FIXME: Fix both unconmmented cases
# yaml_fail should actually fail
# sample_different_factory: returns a zero length straight that gives an error
# when extracting the netlist

yaml_strings = dict(
    yaml_anchor=yaml_anchor,
    # yaml_fail=yaml_fail,
    # sample_regex_connections_backwards=sample_regex_connections_backwards,
    # sample_regex_connections=sample_regex_connections,
    sample_docstring=sample_docstring,
    sample_waypoints=sample_waypoints,
    sample_different_link_factory=sample_different_link_factory,
    # sample_different_factory=sample_different_factory,
    sample_mirror_simple=sample_mirror_simple,
    sample_connections=sample_connections,
    sample_mmis=sample_mmis,
)


@pytest.mark.parametrize("yaml_key", yaml_strings.keys())
def test_gds(yaml_key: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    yaml_string = yaml_strings[yaml_key]
    c = component_from_yaml(yaml_string)
    difftest(c)


@pytest.mark.parametrize("yaml_key", yaml_strings.keys())
def test_settings(
    yaml_key: str, data_regression: DataRegressionFixture, check: bool = True
) -> Component:
    """Avoid regressions when exporting settings."""
    yaml_string = yaml_strings[yaml_key]
    c = component_from_yaml(yaml_string)

    settings = c.get_settings()
    # routes = settings.get("info", {}).get("routes", {})
    # data_regression.check(routes)
    if check:
        data_regression.check(settings)
    return c


@pytest.mark.parametrize("yaml_key", yaml_strings.keys())
def test_ports(yaml_key: str, num_regression: NumericRegressionFixture) -> None:
    """Avoid regressions in port names and locations."""
    yaml_string = yaml_strings[yaml_key]
    c = component_from_yaml(yaml_string)
    if c.ports:
        num_regression.check(c.get_ports_array())


@pytest.mark.parametrize(
    "yaml_key,full_settings", it.product(yaml_strings.keys(), [True, False])
)
def test_netlists(
    yaml_key: str,
    full_settings: bool,
    data_regression: DataRegressionFixture,
    check: bool = True,
) -> None:
    """Write netlists for hierarchical circuits.
    Checks that both netlists are the same
    jsondiff does a hierarchical diff

    Component -> netlist -> Component -> netlist
    """
    yaml_string = yaml_strings[yaml_key]
    c = component_from_yaml(yaml_string)
    n = c.get_netlist(full_settings=full_settings)
    if check:
        data_regression.check(n)

    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    # print(yaml_str)
    c2 = component_from_yaml(yaml_str)
    n2 = c2.get_netlist(full_settings=full_settings)
    d = jsondiff.diff(n, n2)
    assert len(d) == 0, print(d)
    return c2


def _demo_netlist():
    """: there is a zero length path on the route"""
    import gdsfactory as gf

    # c = component_from_yaml(sample_2x2_connections)
    c = component_from_yaml(sample_waypoints)
    c = component_from_yaml(sample_different_factory)
    c.show()
    full_settings = True
    n = c.get_netlist(full_settings=full_settings)
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = component_from_yaml(yaml_str)
    n2 = c2.get_netlist(full_settings=full_settings)
    d = jsondiff.diff(n, n2)
    assert len(d) == 0
    gf.show(c2)


if __name__ == "__main__":
    c = test_sample()
    # c = test_netlists("sample_different_link_factory", True, None, check=False)
    # c = test_netlists("sample_mmis", True, None, check=False)
    # c = test_connections_regex_backwargs()
    # c = test_mirror()
    # c = test_connections()
    # c = test_sample()
    # c = test_connections_2x2()
    # c = test_connections_different_factory()
    # c = test_connections_different_link_factory()
    # c = test_connections_regex()
    # c = test_connections_waypoints()
    # c = test_docstring_sample()
    # c = test_settings("yaml_anchor", None, False)
    # c = test_netlists("yaml_anchor", True, None, False)
    # c = test_netlists("sample_waypoints", True, None, False)

    # c = component_from_yaml(sample_docstring)
    # c = component_from_yaml(sample_different_link_factory)
    c = component_from_yaml(sample_mirror_simple)
    c.show()
