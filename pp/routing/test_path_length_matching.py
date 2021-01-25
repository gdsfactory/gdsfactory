import numpy as np

import pp
from pp.component import Component


def test_path_length_matching() -> Component:
    c = pp.Component("path_length_match_sample")

    dy = 2000.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 100.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N)]

    a1 = 90
    a2 = a1 + 180

    ports1 = [pp.Port(f"top_{i}", (xs1[i], 0), 0.5, a1) for i in range(N)]
    ports2 = [pp.Port(f"bottom_{i}", (xs2[i], dy), 0.5, a2) for i in range(N)]

    routes = pp.routing.connect_bundle_path_length_match(ports1, ports2)
    lengths = [2656.248]
    for route, length in zip(routes, lengths):
        print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)
    return c


def test_path_length_matching_extra_length() -> Component:
    c = pp.Component("path_length_match_sample")

    dy = 2000.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 100.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N)]

    a1 = 90
    a2 = a1 + 180

    ports1 = [pp.Port(f"top_{i}", (xs1[i], 0), 0.5, a1) for i in range(N)]
    ports2 = [pp.Port(f"bottom_{i}", (xs2[i], dy), 0.5, a2) for i in range(N)]

    routes = pp.routing.connect_bundle_path_length_match(
        ports1, ports2, extra_length=40
    )
    lengths = [2656.248 + 40]
    for route, length in zip(routes, lengths):
        print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)
    return c


def test_path_length_matching_nb_loops() -> Component:
    c = pp.Component("path_length_match_sample")

    dy = 2000.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 100.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N)]

    a1 = 90
    a2 = a1 + 180

    ports1 = [pp.Port(f"top_{i}", (xs1[i], 0), 0.5, a1) for i in range(N)]
    ports2 = [pp.Port(f"bottom_{i}", (xs2[i], dy), 0.5, a2) for i in range(N)]

    routes = pp.routing.connect_bundle_path_length_match(ports1, ports2, nb_loops=2)
    lengths = [2681.080]
    for route, length in zip(routes, lengths):
        print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)
    return c


if __name__ == "__main__":
    c = test_path_length_matching()
    # c = test_path_length_matching_extra_length()
    # c = test_path_length_matching_nb_loops()
    pp.show(c)
