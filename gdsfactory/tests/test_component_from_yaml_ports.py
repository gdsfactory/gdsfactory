from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf

yaml_str = """
name: component_yaml_ports

ports:
  o1:
    name: o1
    center:
        - -0.5
        - 0.225
    width: 0.45
    orientation: 180
    port_type: optical
    layer: WG
  o2:
    name: o2
    center:
        - 13
        - 3.675
    width: 0.45
    orientation: 0
    port_type: optical
    layer: [1, 0]
  o3:
    name: o3
    center:
        - 13
        - -3.225
    width: 0.45
    orientation: 0
    port_type: optical
    layer: [1, 0]

instances:
    mmi_bot:
      component: compass
      settings:
        port_type: None
        size: [13.5, 7]

"""


def test_component_from_yaml_ports(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    if check:
        c = gf.read.from_yaml(yaml_str)
        data_regression.check(c.to_dict())


if __name__ == "__main__":
    # test_component_from_yaml_ports(None, check=False)
    c = gf.read.from_yaml(yaml_str)
