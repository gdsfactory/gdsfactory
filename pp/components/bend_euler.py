from typing import Iterable, Optional, Tuple, Union

import numpy as np

from pp.cell import cell
from pp.component import Component
from pp.components.bend_euler_points import euler_bend_points, euler_length
from pp.config import conf
from pp.geo_utils import extrude_path
from pp.layers import LAYER
from pp.port import auto_rename_ports


@cell
def bend_euler(
    theta: int = 90,
    radius: Union[int, float] = 10.0,
    width: float = 0.5,
    resolution: float = 150.0,
    layer: Tuple[int, int] = LAYER.WG,
    layers_cladding: Optional[Iterable[Tuple[int, int]]] = None,
    cladding_offset: float = conf.tech.cladding_offset,
) -> Component:
    """Returns an euler bend.

    Args:
        theta: angle of arc (degrees)
        radius
        width: of the waveguide
        resolution: number of points per theta
        layer
        layers_cladding
        cladding_offset: of layers_cladding

    .. plot::
      :include-source:

      import pp

      c = pp.c.bend_euler()
      c.plot()

    """
    c = Component()
    backbone = euler_bend_points(theta, radius=radius, resolution=resolution)

    pts = extrude_path(backbone, width)
    c.add_polygon(pts, layer=layer)

    layers_cladding = layers_cladding or []
    for layers_cladding in layers_cladding:
        pts = extrude_path(backbone, width + 2 * cladding_offset)
        c.add_polygon(pts, layer=layers_cladding)

    length = euler_length(radius, theta)
    c.info["length"] = length
    c.length = length
    c.radius = radius
    c.add_port(
        name="W0",
        midpoint=np.round(backbone[0].xy, 3),
        orientation=180,
        layer=layer,
        width=width,
    )
    c.add_port(
        name="N0",
        midpoint=np.round(backbone[-1].xy, 3),
        orientation=theta,
        layer=layer,
        width=width,
    )

    return auto_rename_ports(c)


@cell
def bend_euler180(
    radius: Union[int, float] = 10.0,
    width: float = 0.5,
    resolution: float = 150.0,
    layer: Tuple[int, int] = LAYER.WG,
    layers_cladding: Optional[Iterable[Tuple[int, int]]] = None,
    cladding_offset: float = conf.tech.cladding_offset,
) -> Component:
    """
    .. plot::
      :include-source:

      import pp

      c = pp.c.bend_euler180()
      c.plot()

    """
    c = bend_euler(
        theta=180,
        radius=radius,
        width=width,
        resolution=resolution,
        layer=layer,
        layers_cladding=layers_cladding,
        cladding_offset=cladding_offset,
    )
    return auto_rename_ports(c)


if __name__ == "__main__":
    c = bend_euler(layers_cladding=(LAYER.WGCLAD,), cladding_offset=2.0)
    # c = bend_euler90_biased()
    # c = bend_euler180()
    print(c.ports)
    c.show()
