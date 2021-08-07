import gdsfactory as gf
from gdsfactory.difftest import difftest
from gdsfactory.routing.routing import route_manhattan


def test_route_manhattan_circular():
    c = gf.Component("test_route_manhattan_circular")
    pitch = 9.0
    ys1 = [0, 10, 20]
    N = len(ys1)
    ys2 = [15 + i * pitch for i in range(N)]

    ports1 = [gf.Port(f"L_{i}", (0, ys1[i]), 0.5, 0) for i in range(N)]
    ports2 = [gf.Port(f"R_{i}", (20, ys2[i]), 0.5, 180) for i in range(N)]

    for i in range(N):
        route = route_manhattan(ports1[i], ports2[i], radius=5, bendType="circular")
        # references = route_basic(port1=ports1[i], port2=ports2[i])
        c.add(route.references)
    print(route.length)
    assert route.length == 28.708
    difftest(c)
    return c


if __name__ == "__main__":
    c = test_route_manhattan_circular()
    c.show()
