import jsondiff
import numpy as np
import pytest
from omegaconf import OmegaConf
from pytest_regressions.data_regression import DataRegressionFixture

from gdsfactory.component import Component
from gdsfactory.difftest import difftest
from gdsfactory.read.from_yaml import from_yaml, sample_doe_function, sample_mmis

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
    wgw,o1: wgn,o2

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
    s:
        component: straight

    b:
        component: bend_circular

placements:
    b:
        mirror: True
        port: o1

connections:
    b,o1: s,o2

"""


def test_sample() -> Component:
    c = from_yaml(sample_mmis)
    assert len(c.get_dependencies()) == 6, len(c.get_dependencies())
    assert len(c.ports) == 3, len(c.ports)
    return c


def test_connections() -> Component:
    c = from_yaml(sample_connections)
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
            mmi_bottom,o4: mmi_top,o1
            mmi_bottom,o3: mmi_top,o2

        settings:
            cross_section:
                cross_section: strip
                settings:
                    layer: [2, 0]

"""


def test_connections_2x2() -> Component:
    c = from_yaml(sample_2x2_connections)
    assert len(c.get_dependencies()) == 8, len(c.get_dependencies())
    assert len(c.ports) == 0, len(c.ports)

    length = c.routes["mmi_bottom,o3:mmi_top,o2"]
    assert np.isclose(length, 168.274), length
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
            separation: 20
            cross_section:
                cross_section: metal3_with_bend
                settings:
                    layer: [31, 0]
                    width: 10
        links:
            tl,e3: tr,e1
            bl,e3: br,e1
    optical:
        settings:
            cross_section:
                cross_section: strip
                settings:
                    radius: 100
        links:
            bl,e4: br,e3

"""


def test_connections_different_factory() -> Component:
    c = from_yaml(sample_different_factory)
    lengths = [693.274, 693.274, 1199.144]

    assert np.isclose(c.routes["tl,e3:tr,e1"], lengths[0]), c.routes["tl,e3:tr,e1"]
    assert np.isclose(c.routes["bl,e3:br,e1"], lengths[1]), c.routes["bl,e3:br,e1"]
    assert np.isclose(c.routes["bl,e4:br,e3"], lengths[2]), c.routes["bl,e4:br,e3"]

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
            tl,e3: tr,e1
            bl,e3: br,e1

"""


def test_connections_different_link_factory() -> Component:
    c = from_yaml(sample_different_link_factory)

    length = 1719.822
    assert np.isclose(c.routes["tl,e3:tr,e1"], length), c.routes["tl,e3:tr,e1"]
    assert np.isclose(c.routes["bl,e3:br,e1"], length), c.routes["bl,e3:br,e1"]
    return c


sample_waypoints = """
name: sample_waypoints

instances:
    t:
      component: pad_array
      settings:
          orientation: 270
    b:
      component: pad_array
      settings:
          orientation: 90

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
            b,e11: t,e11
            b,e12: t,e12
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
        port: o1
        x: 0
        y: 0
    mmi_bot:
        port: o1
        x: mmi_top,o2
        y: mmi_top,o2
        dx: 40
        dy: -40
routes:
    optical:
        links:
            mmi_top,o3: mmi_bot,o1
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
            left,o:1:3: right,o:1:3
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
            left,o:3:1: right,o:3:1
"""


def test_connections_regex() -> Component:
    c = from_yaml(sample_regex_connections)
    route_names = ["left,o1:right,o1", "left,o2:right,o2", "left,o3:right,o3"]

    length = 12.0
    for route_name in route_names:
        assert np.isclose(c.routes[route_name], length)
    return c


def test_connections_regex_backwargs() -> Component:
    c = from_yaml(sample_regex_connections_backwards)
    route_names = ["left,o1:right,o1", "left,o2:right,o2", "left,o3:right,o3"]

    length = 12.0
    for route_name in route_names:
        assert np.isclose(c.routes[route_name], length), c.routes[route_name]
    return c


def test_connections_waypoints() -> Component:
    c = from_yaml(sample_waypoints)

    length = 2036.548
    route_name = "b,e11:t,e11"
    assert np.isclose(c.routes[route_name], length), c.routes[route_name]
    return c


def test_docstring_sample() -> Component:
    c = from_yaml(sample_docstring)
    route_name = "mmi_top,o3:mmi_bot,o1"
    length = 72.024
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
        port: o1
        x: mmi_long,o2
        y: mmi_long,o2
    mmi_long:
        port: o1
        x: mmi_short,o2
        y: mmi_short,o2
        dx : 10
        dy: 20
"""
#                      ______
#                     |      |
#           dx   1----| short|
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
        port: o3
        x: 0
        y: 0
    mmi_long:
        port: o1
        x: mmi_short,east
        y: mmi_short,north
        dx : 10
        dy: 10
"""

sample_doe = """
name: mask

instances:
    mmi1x2_sweep:
       component: pack_doe
       settings:
         doe: mmi1x2
         do_permutations: True
         spacing: 100
         settings:
           length_mmi: [2, 100]
           width_mmi: [4, 10]
"""

sample_doe_grid = """
name: mask_grid

instances:
    mmi1x2_sweep:
       component: pack_doe_grid
       settings:
         doe: mmi1x2
         do_permutations: True
         spacing: [100, 100]
         shape: [2, 2]
         settings:
           length_mmi: [2, 100]
           width_mmi: [4, 10]
"""

sample_rotation = """
name: sample_rotation

instances:
  r1:
    component: rectangle
    settings:
        size: [4, 2]
  r2:
    component: rectangle
    settings:
        size: [2, 4]

placements:
    r1:
        xmin: 0
        ymin: 0
    r2:
        rotation: -90
        xmin: r1,east
        ymin: 0

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
    sample_doe=sample_doe,
    # sample_doe_grid=sample_doe_grid,
    sample_doe_function=sample_doe_function,
    sample_rotation=sample_rotation,
)


@pytest.mark.parametrize("yaml_key", yaml_strings.keys())
def test_gds(yaml_key: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    yaml_string = yaml_strings[yaml_key]
    c = from_yaml(yaml_string)
    difftest(c)


@pytest.mark.parametrize("yaml_key", yaml_strings.keys())
def test_settings(
    yaml_key: str, data_regression: DataRegressionFixture, check: bool = True
) -> Component:
    """Avoid regressions when exporting settings."""
    yaml_string = yaml_strings[yaml_key]
    c = from_yaml(yaml_string)

    if check:
        data_regression.check(c.to_dict())
    return c


@pytest.mark.parametrize("yaml_key", yaml_strings.keys())
def test_netlists(
    yaml_key: str,
    data_regression: DataRegressionFixture,
    check: bool = True,
) -> None:
    """Write netlists for hierarchical circuits. Checks that both netlists are
    the same jsondiff does a hierarchical diff Component -> netlist ->
    Component -> netlist.

    Args:
        yaml_key: to test.
        data_regression: for regression test.
        check: False, skips test.

    """
    yaml_string = yaml_strings[yaml_key]
    c = from_yaml(yaml_string)
    n = c.get_netlist()
    if check:
        data_regression.check(n)

    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)

    # print(yaml_str)
    c2 = from_yaml(yaml_str, name=c.name)
    n2 = c2.get_netlist()
    # pprint(d)
    # assert len(d) == 0, pprint(d)
    return jsondiff.diff(n, n2)


def _demo_netlist() -> None:
    """path on the route."""
    import gdsfactory as gf

    # c = from_yaml(sample_2x2_connections)
    c = from_yaml(sample_waypoints)
    c = from_yaml(sample_different_factory)
    c.show(show_ports=True)
    n = c.get_netlist()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = from_yaml(yaml_str)
    n2 = c2.get_netlist()
    d = jsondiff.diff(n, n2)
    assert len(d) == 0
    gf.show(c2)


if __name__ == "__main__":
    c = test_connections_different_factory()
    c.show()

    # c = test_netlists("sample_mmis", None, False)
    # yaml_key = "sample_doe_function"
    # yaml_key = "sample_mmis"
    # yaml_string = yaml_strings[yaml_key]
    # c = from_yaml(yaml_string)
    # n = c.get_netlist()
    # yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    # c2 = from_yaml(yaml_str)
    # n2 = c2.get_netlist()
    # d = jsondiff.diff(n, n2)
    # pprint(d)
    # c2.show()
