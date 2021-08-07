import gdsfactory as gf

yaml = """
connections:
  bend_left,N0: straight_left,W0
  bend_right,N0: straight_top,E0
  bend_right,W0: straight_right,W0
  coupler_ring_edf8f53f,N0: straight_left,E0
  coupler_ring_edf8f53f,N1: straight_right,E0
  straight_top,W0: bend_left,W0
instances:
  bend_left:
    component: bend_euler
    settings:
      radius: 10.0
      width: 0.5
  bend_right:
    component: bend_euler
    settings:
      radius: 10.0
      width: 0.5
  coupler_ring_edf8f53f:
    component: coupler_ring
    settings:
      radius: 10.0
      gap: 0.2
      length_x: 4.0
      width: 0.5
  straight_left:
    component: straight
    settings:
      length: 0.001
      width: 0.5
  straight_right:
    component: straight
    settings:
      length: 0.001
      width: 0.5
  straight_top:
    component: straight
    settings:
      length: 4.0
      width: 0.5
placements:
  straight_left:
    rotation: 270
    x: -14.0
    'y': 10.7
  straight_right:
    rotation: 270
    x: 10.0
    'y': 10.7
ports:
  E0: coupler_ring_edf8f53f,E0
  W0: coupler_ring_edf8f53f,W0
"""


def test_netlist_labels() -> None:
    c = gf.component_from_yaml(yaml)
    n = c.get_netlist()
    placements = n["placements"]
    print(placements)
    assert "bend_left" in placements
    assert "bend_right" in placements


if __name__ == "__main__":
    c = gf.component_from_yaml(yaml)
    n = c.get_netlist()
    print(n["placements"])
    c.show()
