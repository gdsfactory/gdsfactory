import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.difftest import difftest

mirror_port = """
name: mirror_port
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_long:
        port: o1
        x: 20
        y: 10
        mirror: True

ports:
    o1: mmi_long,o3
    o2: mmi_long,o2
    o3: mmi_long,o1
"""


mirror_x = """
name: mirror_x
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
    o1: mmi_long,o3
    o2: mmi_long,o2
    o3: mmi_long,o1
"""


rotation = """
name: rotation
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_long:
        port: o1
        x: 10
        y: 20
        rotation: 90
ports:
    o1: mmi_long,o3
    o2: mmi_long,o2
    o3: mmi_long,o1
"""

dxdy = """
name: dxdy
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
        x: 0
        y: 0
    mmi_long:
        port: o1
        x: mmi_short,o2
        y: mmi_short,o2
        dx: 10
        dy: -10
ports:
    o1: mmi_long,o3
"""


yaml_list = [mirror_port, mirror_x, rotation, dxdy]


@pytest.mark.parametrize("yaml_index", range(len(yaml_list)))
def test_components(
    yaml_index: int, data_regression: DataRegressionFixture, check: bool = True
) -> None:
    yaml = yaml_list[yaml_index]
    c = gf.read.from_yaml(yaml)
    difftest(c)
    if check:
        data_regression.check(c.to_dict())


if __name__ == "__main__":
    c = gf.read.from_yaml(mirror_port)
    c = gf.read.from_yaml(dxdy)
    c.show()
