import pytest
from pytest_regressions.num_regression import NumericRegressionFixture

import pp

mirror_port = """
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_long:
        port: W0
        x: 20
        y: 10
        mirror: True

ports:
    W0: mmi_long,E0
    W1: mmi_long,E1
    E0: mmi_long,W0
"""


mirror_x = """
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_long:
        x: 0
        y: 0
        mirror: 25
ports:
    W0: mmi_long,E0
    W1: mmi_long,E1
    E0: mmi_long,W0
"""


rotation = """
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_long:
        port: W0
        x: 10
        y: 20
        rotation: 90
ports:
    N0: mmi_long,E0
    N1: mmi_long,E1
    S0: mmi_long,W0
"""

dxdy = """
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
        x: 0
        y: 0
    mmi_long:
        port: W0
        x: mmi_short,E1
        y: mmi_short,E1
        dx: 10
        dy: -10
ports:
    W0: mmi_long,E0
"""


yaml_list = [mirror_port, mirror_x, rotation, dxdy]


@pytest.mark.parametrize("yaml_index", range(len(yaml_list)))
def test_components_ports(
    yaml_index: int, num_regression: NumericRegressionFixture
) -> None:
    yaml = yaml_list[yaml_index]
    c = pp.component_from_yaml(yaml)
    if c.ports:
        num_regression.check(c.get_ports_array())


if __name__ == "__main__":
    c = pp.component_from_yaml(mirror_port)
    pp.show(c)
