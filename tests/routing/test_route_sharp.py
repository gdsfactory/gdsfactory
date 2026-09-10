import numpy as np

import gdsfactory as gf
from gdsfactory.routing.route_sharp import path_V


def test_path_v_intersection() -> None:
    port1 = gf.Port(
        "p1",
        center=(0, 0),
        cross_section=gf.cross_section.strip(width=0.5),
        orientation=0,
    )
    port2 = gf.Port(
        "p2",
        center=(10, -10),
        cross_section=gf.cross_section.strip(width=0.5),
        orientation=90,
    )

    path = path_V(port1, port2)

    np.testing.assert_allclose(path.points, ((0, 0), (10, 0), (10, -10)))


def test_route_sharp_tapers_between_port_widths() -> None:
    c = gf.Component()
    p1 = c.add_port(
        "in", center=(0, 0), orientation=0, cross_section=gf.cross_section.strip()
    )
    p2 = c.add_port(
        "out",
        center=(10, 0),
        orientation=180,
        cross_section=gf.cross_section.strip(width=1),
    )
    gf.routing.route_sharp(c, p1, p2, path_type="straight")
    assert c.area("WG") == 7.5
