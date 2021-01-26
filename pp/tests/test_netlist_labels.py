import pp

yaml = """
connections:
  bend_left,N0: waveguide_left,W0
  bend_right,N0: waveguide_top,E0
  bend_right,W0: waveguide_right,W0
  coupler_ring_edf8f53f,N0: waveguide_left,E0
  coupler_ring_edf8f53f,N1: waveguide_right,E0
  waveguide_top,W0: bend_left,W0
instances:
  bend_left:
    component: bend_circular
    settings:
      radius: 10.0
      width: 0.5
  bend_right:
    component: bend_circular
    settings:
      radius: 10.0
      width: 0.5
  coupler_ring_edf8f53f:
    component: coupler_ring
    settings:
      bend_radius: 10.0
      gap: 0.2
      length_x: 4.0
      wg_width: 0.5
  waveguide_left:
    component: waveguide
    settings:
      length: 0.001
      width: 0.5
  waveguide_right:
    component: waveguide
    settings:
      length: 0.001
      width: 0.5
  waveguide_top:
    component: waveguide
    settings:
      length: 4.0
      width: 0.5
placements:
  waveguide_left:
    rotation: 270
    x: -14.0
    'y': 10.7
  waveguide_right:
    rotation: 270
    x: 10.0
    'y': 10.7
ports:
  E0: coupler_ring_edf8f53f,E0
  W0: coupler_ring_edf8f53f,W0
"""


def test_netlist_labels() -> None:
    c = pp.component_from_yaml(yaml)
    n = c.get_netlist()
    placements = n["placements"]
    print(placements)
    assert "bend_left" in placements
    assert "bend_right" in placements


if __name__ == "__main__":
    c = pp.component_from_yaml(yaml)
    n = c.get_netlist()
    print(n["placements"])
    pp.show(c)
