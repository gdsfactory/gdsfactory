import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

yaml_str = """
instances:
  t:
    component: pad_array
    settings:
      port_orientation: 270
      columns: 3
  b:
    component: pad_array
    settings:
      port_orientation: 90
      columns: 3

placements:
  t:
    x: 200
    y: 400
routes:
  route1:
    settings:
      bend: wire_corner
      start_straight_length: 150
      end_straight_length: 150
      cross_section: metal_routing
      allow_width_mismatch: True
      sort_ports: True
    links:
      t,e11: b,e11
      t,e13: b,e13
"""

if __name__ == "__main__":
    c = gf.read.from_yaml(yaml_str)
    c.show()
