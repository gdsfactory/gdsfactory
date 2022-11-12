from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component


def test_get_bundle_sort_ports(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    lengths = {}
    c = gf.Component("test_get_bundle_sort_ports")
    ys_right = [0, 10, 20, 40, 50, 80]
    pitch = 127.0
    N = len(ys_right)
    ys_left = [(i - N / 2) * pitch for i in range(N)]
    layer = (1, 0)

    right_ports = [
        gf.Port(
            f"R_{i}", center=(0, ys_right[i]), width=0.5, orientation=180, layer=layer
        )
        for i in range(N)
    ]
    left_ports = [
        gf.Port(
            f"L_{i}", center=(-400, ys_left[i]), width=0.5, orientation=0, layer=layer
        )
        for i in range(N)
    ]
    left_ports.reverse()
    routes = gf.routing.get_bundle(right_ports, left_ports)

    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_get_bundle_sort_ports(None, check=False)
    c.show(show_ports=True)
