from __future__ import annotations

import numpy as np

import gdsfactory as gf


def test_get_bundle_path_length_matching() -> None:
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
        assert np.isclose(route.length, length), route.length


def test_get_bundle_path_length_matching_extra_length() -> None:
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
        assert np.isclose(route.length, length), route.length


def test_get_bundle_path_length_matching_nb_loops() -> None:
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
        assert np.isclose(route.length, length), route.length


if __name__ == "__main__":
    test_get_bundle_path_length_matching()
    # test_get_bundle_path_length_matching_extra_length()
    # test_get_bundle_path_length_matching_nb_loops()
