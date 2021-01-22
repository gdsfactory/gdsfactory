import numpy as np
import pytest

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


def test_sample():
    c = component_from_yaml(sample_mmis)
    assert len(c.get_dependencies()) == 3
    assert len(c.ports) == 2
    return c


def test_connections():
    c = component_from_yaml(sample_connections)
    # print(len(c.get_dependencies()))
    # print(len(c.ports))
    assert len(c.get_dependencies()) == 2
    assert len(c.ports) == 0
    return c


def test_mirror():
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


def test_connections_2x2():
    c = component_from_yaml(sample_2x2_connections)
    print(len(c.get_dependencies()))
    print(len(c.ports))
    assert len(c.get_dependencies()) == 4
    assert len(c.ports) == 0
    length = c.routes["mmi_bottom,E1:mmi_top,W1"].parent.length
    print(length)
    assert np.isclose(length, 163.91592653589794)
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


def test_connections_different_factory():
    c = component_from_yaml(sample_different_factory)
    # print(c.routes["bl,S:br,E"].parent.length)
    assert np.isclose(c.routes["tl,E:tr,W"].parent.length, 700.0)
    assert np.isclose(c.routes["bl,E:br,W"].parent.length, 850.0)
    assert np.isclose(c.routes["bl,S:br,E"].parent.length, 1171.258898038469)
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


def test_connections_different_link_factory():
    c = component_from_yaml(sample_different_link_factory)
    # print(c.routes['tl,E:tr,W'].parent.length)
    # print(c.routes['bl,E:br,W'].parent.length)

    length = 1716.2477796076937
    assert np.isclose(c.routes["tl,E:tr,W"].parent.length, length)
    assert np.isclose(c.routes["bl,E:br,W"].parent.length, length)
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


def test_connections_regex():
    c = component_from_yaml(sample_regex_connections)
    route_names = ["left,E0:right,W0", "left,E1:right,W1", "left,E2:right,W2"]

    length = 12.0
    for route_name in route_names:
        print(c.routes[route_name].parent.length)
        assert np.isclose(c.routes[route_name].parent.length, length)
    return c


def test_connections_regex_backwargs():
    c = component_from_yaml(sample_regex_connections_backwards)
    route_names = ["left,E0:right,W0", "left,E1:right,W1", "left,E2:right,W2"]

    length = 12.0
    for route_name in route_names:
        print(c.routes[route_name].parent.length)
        assert np.isclose(c.routes[route_name].parent.length, length)
    return c


def test_connections_waypoints():
    c = component_from_yaml(sample_waypoints)
    # print(c.routes['t,S5:b,N4'].parent.length)

    length = 1241.415926535898
    assert np.isclose(c.routes["t,S5:b,N4"].parent.length, length)
    return c


def test_docstring_sample():
    c = component_from_yaml(sample_docstring)
    route_name = "mmi_top,E0:mmi_bot,W0"
    length = 50.16592653589793
    # print(c.routes[route_name].parent.length)
    assert np.isclose(c.routes[route_name].parent.length, length)
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

yaml_strings = dict(
    yaml_anchor=yaml_anchor,
    yaml_fail=yaml_fail,
    sample_regex_connections_backwards=sample_regex_connections_backwards,
    sample_regex_connections=sample_regex_connections,
    sample_docstring=sample_docstring,
    sample_waypoints=sample_waypoints,
    sample_different_link_factory=sample_different_link_factory,
    sample_different_factory=sample_different_factory,
    sample_mirror_simple=sample_mirror_simple,
    sample_connections=sample_connections,
    sample_mmis=sample_mmis,
)


@pytest.mark.parametrize("yaml_key", yaml_strings.keys())
def test_gds(yaml_key, data_regression):
    """Avoid regressions in GDS geometry shapes and layers."""
    yaml_string = yaml_strings[yaml_key]
    c = component_from_yaml(yaml_string)
    difftest(c)


@pytest.mark.parametrize("yaml_key", yaml_strings.keys())
def test_settings(yaml_key, data_regression):
    """Avoid regressions when exporting settings."""
    yaml_string = yaml_strings[yaml_key]
    c = component_from_yaml(yaml_string)
    settings = c.get_settings()
    data_regression.check(settings)


@pytest.mark.parametrize("yaml_key", yaml_strings.keys())
def test_ports(yaml_key, num_regression):
    """Avoid regressions in port names and locations."""
    yaml_string = yaml_strings[yaml_key]
    c = component_from_yaml(yaml_string)
    if c.ports:
        num_regression.check(c.get_ports_array())


if __name__ == "__main__":
    c = component_from_yaml(sample_connections)
