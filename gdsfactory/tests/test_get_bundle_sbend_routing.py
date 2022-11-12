from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component


def demo_get_bundle_sbend_routing(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:
    """FIXME."""

    lengths = {}

    c = gf.Component("test_get_bundle_sort_ports")
    pitch = 2.0
    ys_left = [0, 10, 20]
    N = len(ys_left)
    ys_right = [(i - N / 2) * pitch for i in range(N)]

    right_ports = [
        gf.Port(f"R_{i}", center=(0, ys_right[i]), width=0.5, orientation=180)
        for i in range(N)
    ]
    left_ports = [
        gf.Port(f"L_{i}", center=(-50, ys_left[i]), width=0.5, orientation=0)
        for i in range(N)
    ]
    left_ports.reverse()
    routes = gf.routing.get_bundle(right_ports, left_ports, bend_radius=5)

    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = demo_get_bundle_sbend_routing(None, check=False)
    c.show(show_ports=True)
