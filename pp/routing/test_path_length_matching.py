import numpy as np
import pp


def test_path_length_matching():
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
    for route in routes:
        # print(route.parent.length)
        assert np.isclose(route.parent.length, 2656.2477796076937)
    c.add(routes)
    c.routes = routes
    return c


def test_path_length_matching_extra_length():
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
    for route in routes:
        # print(route.parent.length)
        assert np.isclose(route.parent.length, 2656.2477796076937 + 40)
    c.add(routes)
    c.routes = routes
    return c


def test_path_length_matching_nb_loops():
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
    for route in routes:
        # print(route.parent.length)
        assert np.isclose(route.parent.length, 2681.07963267949)
    c.add(routes)
    c.routes = routes
    return c


if __name__ == "__main__":
    # c = test_path_length_matching()
    # c = test_path_length_matching_extra_length()
    c = test_path_length_matching_nb_loops()
    pp.show(c)
