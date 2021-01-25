import itertools as it

import jsondiff
import numpy as np
import pytest
from omegaconf import OmegaConf
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

from pp.component import Component
from pp.component_from_yaml import component_from_yaml, sample_mmis
from pp.testing import difftest

sample_connections = """
name: sample_connections

instances:
    wgw:
      component: waveguide
      settings:
        width: 1
        length: 1
    wgn:
      component: waveguide
      settings:
        width: 0.5
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

sample_mirror = """
name: mzi_with_mirrored_arm

instances:
    CP1:
      component: mmi1x2
      settings:
          width_mmi: 4.5
          length_mmi: 10
    CP2:
        component: mmi1x2
        settings:
            width_mmi: 4.5
            length_mmi: 5
    arm_top:
        component: mzi_arm
        settings:
            L0: 30
    arm_bot:
        component: mzi_arm
        settings:
            L0: 15

placements:
    arm_bot:
        port: E0
        mirror: True

ports:
    W0: CP1,W0
    E0: CP2,W0

connections:
    arm_bot,W0: CP1,E0
    arm_top,W0: CP1,E1
    CP2,E0: arm_bot,E0
    CP2,E1: arm_top,E0
"""


sample_mirror_simple = """
name: sample_mirror_simple

instances:
    w:
        component: waveguide

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
    # print(len(c.get_dependencies()))
    # print(len(c.ports))
    assert len(c.get_dependencies()) == 7
    assert len(c.ports) == 2
    return c


def test_connections() -> Component:
    c = component_from_yaml(sample_connections)
    # print(len(c.get_dependencies()))
    # print(len(c.ports))
    assert len(c.get_dependencies()) == 2
    assert len(c.ports) == 0
    return c


def test_mirror() -> Component:
    c = component_from_yaml(sample_mirror)
    # print(len(c.get_dependencies()))
    # print(len(c.ports))
    assert len(c.get_dependencies()) == 4
    assert len(c.ports) == 2
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
        factory: optical
        links:
            mmi_bottom,E0: mmi_top,W0
            mmi_bottom,E1: mmi_top,W1

"""


def test_connections_2x2() -> Component:
    c = component_from_yaml(sample_2x2_connections)
    # print(len(c.get_dependencies()))
    # print(len(c.ports))
    assert len(c.get_dependencies()) == 9
    assert len(c.ports) == 0

    print(c.routes)
    length = c.routes["mmi_bottom,E1:mmi_top,W1"]
    print(length)
    assert length == 163.916
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
        factory: electrical
        settings:
            separation: 240
        links:
            tl,E: tr,W
            bl,E: br,W
    optical:
        factory: optical
        settings:
            bend_radius: 100
        links:
            bl,S: br,E

"""


def test_connections_different_factory() -> Component:
    c = component_from_yaml(sample_different_factory)
    assert np.isclose(c.routes["tl,E:tr,W"], 700.0)
    assert np.isclose(c.routes["bl,E:br,W"], 850.0)
    assert np.isclose(c.routes["bl,S:br,E"], 1171.259)
    return c


sample_different_link_factory = """
name: sample_different_link_factory

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
        factory: optical
        settings:
            bend_radius: 10
        link_factory: link_ports_path_length_match
        link_settings:
            extra_length: 500
        links:
            tl,E: tr,W
            bl,E: br,W

"""


def test_connections_different_link_factory() -> Component:
    c = component_from_yaml(sample_different_link_factory)

    length = 1716.248
    assert np.isclose(c.routes["tl,E:tr,W"], length)
    assert np.isclose(c.routes["bl,E:br,W"], length)
    return c


sample_waypoints = """
name: sample_waypoints

instances:
    t:
      component: pad_array
      settings:
          port_list: ['S']
    b:
      component: pad_array

placements:
    t:
        x: 100
        y: 1000
routes:
    route1:
        factory: optical
        link_factory: link_optical_waypoints
        link_settings:
            way_points:
                - [0,0]
                - [0, 600]
                - [-250, 600]
                - [-250, 1000]
        links:
            t,S5: b,N4
"""


sample_docstring = """
name: sample_docstring

instances:
    mmi_bot:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 10
    mmi_top:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5

placements:
    mmi_top:
        port: W0
        x: 0
        y: 0
    mmi_bot:
        port: W0
        x: mmi_top,E1
        y: mmi_top,E1
        dx: 30
        dy: -30
routes:
    optical:
        factory: optical
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
        factory: optical
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
        factory: optical
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
        print(c.routes[route_name])
        assert np.isclose(c.routes[route_name], length)
    return c


def test_connections_waypoints() -> Component:
    c = component_from_yaml(sample_waypoints)

    length = 1241.415926535898
    route_name = "t,S5:b,N4"
    assert np.isclose(c.routes[route_name], length)
    return c


def test_docstring_sample() -> Component:
    c = component_from_yaml(sample_docstring)
    route_name = "mmi_top,E0:mmi_bot,W0"
    length = 50.16592653589793
    print(c.routes[route_name])
    assert np.isclose(c.routes[route_name], length)
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
# sample_different_factory: returns a zero length waveguide that gives an error
# when extracting the netlist

yaml_strings = dict(
    yaml_anchor=yaml_anchor,
    # yaml_fail=yaml_fail,
    sample_regex_connections_backwards=sample_regex_connections_backwards,
    sample_regex_connections=sample_regex_connections,
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
def test_settings(yaml_key: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    yaml_string = yaml_strings[yaml_key]
    c = component_from_yaml(yaml_string)

    settings = c.get_settings()
    # routes = settings.get("info", {}).get("routes", {})
    # data_regression.check(routes)
    data_regression.check(settings)


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
    yaml_key: str, full_settings: bool, data_regression: DataRegressionFixture
) -> None:
    """Write netlists for hierarchical circuits.
    Checks that both netlists are the same
    jsondiff does a hierarchical diff

    Component -> netlist -> Component -> netlist
    """
    yaml_string = yaml_strings[yaml_key]
    c = component_from_yaml(yaml_string)
    n = c.get_netlist(full_settings=full_settings)
    data_regression.check(n)

    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = component_from_yaml(yaml_str)
    n2 = c2.get_netlist(full_settings=full_settings)
    d = jsondiff.diff(n, n2)
    assert len(d) == 0


def needs_fixing():
    """FIXME: there is a zero length path on the route"""
    import pp

    # c = component_from_yaml(sample_2x2_connections)
    # c = component_from_yaml(sample_waypoints)
    c = component_from_yaml(sample_different_factory)
    pp.show(c)
    full_settings = True
    n = c.get_netlist(full_settings=full_settings)
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = component_from_yaml(yaml_str)
    n2 = c2.get_netlist(full_settings=full_settings)
    d = jsondiff.diff(n, n2)
    assert len(d) == 0
    pp.show(c2)


if __name__ == "__main__":
    import pp

    # c = test_sample()
    # c = test_connections_2x2()
    # c = test_connections_different_factory()
    c = test_connections_regex()
    pp.show(c)
