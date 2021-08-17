import gdsfactory as gf

yaml = """
connections:
  bend_left,2: straight_left,1
  bend_right,2: straight_top,2
  bend_right,1: straight_right,1
  coupler_ring_edf8f53f,2: straight_left,2
  coupler_ring_edf8f53f,3: straight_right,2
  straight_top,1: bend_left,1
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
  4: coupler_ring_edf8f53f,4
  1: coupler_ring_edf8f53f,1
"""


def test_netlist_labels() -> None:
    c = gf.component_from_yaml(yaml)
    n = c.get_netlist()
    placements = n["placements"]
    # print(placements)
    assert "bend_left" in placements
    assert "bend_right" in placements


if __name__ == "__main__":
    c = gf.component_from_yaml(yaml)
    n = c.get_netlist()
    print(n["placements"])
    c.show()
