from pp.difftest import difftest
from pp.routing.route_ports_to_side import sample_route_sides


def test_sample_route_sides() -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    c = sample_route_sides()
    difftest(c)
    return c


if __name__ == "__main__":
    c = test_sample_route_sides()
    c.show()
