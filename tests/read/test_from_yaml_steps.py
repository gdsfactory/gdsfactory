from itertools import pairwise

import pytest

import gdsfactory as gf


YAML_STEPS = """
instances:
  cp1:
    component: coupler
  cp2:
    component: coupler
routes:
  bundle1:
    links:
      cp1,o4: cp2,o1
    routing_strategy: route_bundle
    settings:
      steps:
        - dx: 10
        - dy: -20
        - dx: 10
placements:
  cp1:
    x: 0
    y: 0
    dx: -5.041
    dy: 31.798
  cp2:
    x: 0
    y: 0
    dx: 93.377
    dy: -52.512
"""


def test_from_yaml_error_route_preserves_steps() -> None:
    with pytest.warns(UserWarning, match="Routing failed"):
        component = gf.read.from_yaml(YAML_STEPS)

    route = next(iter(component.routes.values()))
    start = route.backbone[0]
    dbu10 = component.kcl.to_dbu(10)
    dbu20 = component.kcl.to_dbu(20)
    step_points = [
        (start.x + dbu10, start.y),
        (start.x + dbu10, start.y - dbu20),
        (start.x + dbu20, start.y - dbu20),
    ]

    for x, y in step_points:
        assert any(
            min(p1.x, p2.x) <= x <= max(p1.x, p2.x)
            and min(p1.y, p2.y) <= y <= max(p1.y, p2.y)
            for p1, p2 in pairwise(route.backbone)
        )
