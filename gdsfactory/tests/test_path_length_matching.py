import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component


def test_path_length_matching() -> Component:
    c = gf.Component("path_length_match_sample")

    dy = 2000.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 100.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N)]

    a1 = 90
    a2 = a1 + 180

    layer = (1, 0)
    ports1 = [
        gf.Port(f"top_{i}", center=(xs1[i], 0), width=0.5, orientation=a1, layer=layer)
        for i in range(N)
    ]
    ports2 = [
        gf.Port(f"bot_{i}", center=(xs2[i], dy), width=0.5, orientation=a2, layer=layer)
        for i in range(N)
    ]

    routes = gf.routing.get_bundle_path_length_match(ports1, ports2)
    lengths = [2659.822]
    for route, length in zip(routes, lengths):
        c.add(route.references)
        assert np.isclose(route.length, length), route.length
    return c


def test_path_length_matching_extra_length() -> Component:
    c = gf.Component("path_length_match_sample")

    dy = 2000.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 100.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N)]

    a1 = 90
    a2 = a1 + 180
    layer = (1, 0)

    ports1 = [
        gf.Port(f"top_{i}", center=(xs1[i], 0), width=0.5, orientation=a1, layer=layer)
        for i in range(N)
    ]
    ports2 = [
        gf.Port(f"bot_{i}", center=(xs2[i], dy), width=0.5, orientation=a2, layer=layer)
        for i in range(N)
    ]

    routes = gf.routing.get_bundle_path_length_match(ports1, ports2, extra_length=40)
    lengths = [2699.822]
    for route, length in zip(routes, lengths):
        c.add(route.references)
        assert np.isclose(route.length, length), route.length
    return c


def test_path_length_matching_nb_loops() -> Component:
    c = gf.Component("path_length_match_sample")

    dy = 2000.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 100.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N)]

    a1 = 90
    a2 = a1 + 180

    layer = (1, 0)
    ports1 = [
        gf.Port(f"top_{i}", center=(xs1[i], 0), width=0.5, orientation=a1, layer=layer)
        for i in range(N)
    ]
    ports2 = [
        gf.Port(f"bot_{i}", center=(xs2[i], dy), width=0.5, orientation=a2, layer=layer)
        for i in range(N)
    ]

    routes = gf.routing.get_bundle_path_length_match(ports1, ports2, nb_loops=2)
    lengths = [2686.37]
    for route, length in zip(routes, lengths):
        c.add(route.references)
        assert np.isclose(route.length, length), route.length
    return c


if __name__ == "__main__":
    c = test_path_length_matching()
    # c = test_path_length_matching_extra_length()
    # c = test_path_length_matching_nb_loops()
    c.show(show_ports=True)
