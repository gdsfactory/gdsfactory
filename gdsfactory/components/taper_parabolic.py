import numpy as np

import gdsfactory as gf
from gdsfactory.path import transition_exponential
from gdsfactory.types import LayerSpec


@gf.cell
def taper_parabolic(
    length: float = 20,
    width1: float = 0.5,
    width2: float = 5.0,
    exp: float = 0.5,
    npoints: int = 100,
    layer: LayerSpec = "WG",
) -> gf.Component:
    """Returns a parabolic_taper.

    Args:
        length: in um.
        width1: in um.
        width2: in um.
        exp: exponent.
        npoints: number of points.
        layer: layer spec.
    """
    x = np.linspace(0, 1, npoints)
    y = transition_exponential(y1=width1, y2=width2, exp=exp)(x) / 2

    x = length * x
    points1 = np.array([x, y]).T
    points2 = np.flipud(np.array([x, -y]).T)
    points = np.concatenate([points1, points2])

    c = gf.Component()
    c.add_polygon(points, layer=layer)
    c.add_port(name="o1", center=(0, 0), width=width1, orientation=180, layer=layer)
    c.add_port(name="o2", center=(length, 0), width=width2, orientation=0, layer=layer)
    return c


if __name__ == "__main__":
    c = taper_parabolic(width2=6, length=40, exp=0.6)
    c = taper_parabolic()
    c.show(show_ports=True)
