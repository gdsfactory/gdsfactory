import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.routing.get_bundle import get_bundle


def test_get_bundle_u_indirect(angle=0):
    """
    FIXME: start_straight_length is getting ignored in this case

    """
    dy = -200
    xs1 = [-100, -90, -80, -55, -35] + [200, 210, 240]
    axis = "X" if angle in [0, 180] else "Y"

    pitch = 10.0
    N = len(xs1)
    xs2 = [50 + i * pitch for i in range(N)]

    a1 = angle
    a2 = a1 + 180

    if axis == "X":
        ports1 = [Port("top_{}".format(i), (0, xs1[i]), 0.5, a1) for i in range(N)]
        ports2 = [Port("bot_{}".format(i), (dy, xs2[i]), 0.5, a2) for i in range(N)]

    else:
        ports1 = [Port("top_{}".format(i), (xs1[i], 0), 0.5, a1) for i in range(N)]

        ports2 = [Port("bot_{}".format(i), (xs2[i], dy), 0.5, a2) for i in range(N)]

    c = gf.Component(f"test_get_bundle_u_indirect_{angle}_{dy}")
    routes = get_bundle(
        ports1,
        ports2,
        bend=gf.components.bend_circular,
        end_straight_length=5,
        start_straight_length=1,
    )

    for route in routes:
        c.add(route.references)
    return c


if __name__ == "__main__":
    c = test_get_bundle_u_indirect(angle=90)
    c.show()
