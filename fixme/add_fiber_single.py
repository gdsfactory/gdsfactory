"""add fiber single requires to have ports on 2nm grid

"""

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
    mmi:
      component: compass
      settings:
        port_type: None
        size: [13.5, 9]

placements:
    mmi:
        port: e1
        x: -0.5

"""


if __name__ == "__main__":
    # from omegaconf import OmegaConf
    # d = OmegaConf.create(yaml_str)

    c = gf.read.from_yaml(yaml_str)

    cc = gf.routing.add_fiber_single(c, zero_port=None)
    cc.show()
