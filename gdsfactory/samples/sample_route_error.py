"""Routing issue."""

from __future__ import annotations

import gdsfactory as gf

demo = """
instances:
  pad1:
    component: pad
    settings:
      layer: M1
  pad2:
    component: pad
    settings: {}
  pad3:
    component: pad
    settings: {}
  pad4:
    component: pad
    settings: {}
  via:
    component: via1
    settings: {}
connections: {}
routes:
  bundle1:
    links:
      pad1,e3: pad2,e1
    routing_strategy: route_bundle_electrical
    settings:
      auto_taper: false
      allow_layer_mismatch: false
  bundle3:
    links:
      pad3,e3: pad4,e1
    routing_strategy: route_bundle_electrical
    settings:
      auto_taper: false
nets: []
ports: {}
placements:
  pad1:
    x: 0
    y: 0
    dx: 437.67
    dy: 26.62
    rotation: 0
    mirror: false
  pad2:
    dx: -74.566
    dy: 328.078
    rotation: 0
    mirror: false
  pad3:
    x: 0
    y: 0
    dx: 4.157782554626465
    dy: -99.9135284423828
  pad4:
    x: 0
    y: 0
    dx: 632.702
    dy: 352.855
    rotation: 0
    mirror: false
  via:
    x: 100
    y: 230
    dx: 0
    dy: 0
"""

if __name__ == "__main__":
    c = gf.read.from_yaml(demo)
    c.show()
