import gdsfactory as gf

yaml = """
name: test_netlist_labels
connections:
  bend_left,o2: straight_left,o1
  bend_right,o2: straight_top,o2
  bend_right,o1: straight_right,o1
  coupler_ring_edf8f53f,o2: straight_left,o2
  coupler_ring_edf8f53f,o3: straight_right,o2
  straight_top,o1: bend_left,o1
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
  o4: coupler_ring_edf8f53f,o4
  o1: coupler_ring_edf8f53f,o1
"""


def test_netlist_labels() -> None:
    c = gf.read.from_yaml(yaml)
    n = c.get_netlist()
    placements = n["placements"]
    assert "bend_left" in placements, placements
    assert "bend_right" in placements, placements


if __name__ == "__main__":
    c = gf.read.from_yaml(yaml)
    n = c.get_netlist()
    print(n["placements"])
    c.show()
