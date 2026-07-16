import numpy as np

import gdsfactory as gf
from gdsfactory.routing.route_sharp import path_V


def test_path_v_intersection() -> None:
    port1 = gf.Port("p1", center=(0, 0), width=0.5, orientation=0, layer=1)
    port2 = gf.Port("p2", center=(10, -10), width=0.5, orientation=90, layer=1)

    path = path_V(port1, port2)

    np.testing.assert_allclose(path.points, ((0, 0), (10, 0), (10, -10)))
