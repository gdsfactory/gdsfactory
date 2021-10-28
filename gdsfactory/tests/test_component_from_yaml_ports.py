from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf

yaml_str = """

ports:
  o1:
    name: o1
    midpoint:
        - -0.5
        - 0.225
    width: 0.45
    orientation: 180
    port_type: optical
  o2:
    name: o2
    midpoint:
        - 13
        - 3.675
    width: 0.45
    orientation: 0
    port_type: optical
  o3:
    name: o3
    midpoint:
        - 13
        - -3.225
    width: 0.45
    orientation: 0
    port_type: optical

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
    c = gf.read.from_yaml(yaml_str)
    if check:
        data_regression.check(c.to_dict())


if __name__ == "__main__":
    from omegaconf import OmegaConf

    d = OmegaConf.create(yaml_str)
