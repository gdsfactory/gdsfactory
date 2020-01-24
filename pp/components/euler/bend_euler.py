import pp
from pp.geo_utils import extrude_path
from pp.components.euler.geo_euler import euler_bend_points
from pp.components.euler.geo_euler import euler_length
from pp.layers import LAYER
from pp.ports.port_naming import auto_rename_ports
import numpy as np


def _bend_euler(theta=90, radius=10.0, width=0.5, resolution=150.0, layer=LAYER.WG):
    c = pp.Component()
    backbone = euler_bend_points(theta, radius=radius, resolution=resolution)
    pts = extrude_path(backbone, width)

    c.add_polygon(pts, layer=layer)
    # print(backbone[0])
    # print(backbone[-1])
    c.info["length"] = euler_length(radius, theta)
    c.radius = radius
    c.add_port(
        name="in0",
        midpoint=np.round(backbone[0].xy, 3),
        orientation=180,
        layer=layer,
        width=width,
    )
    c.add_port(
        name="out0",
        midpoint=np.round(backbone[-1].xy, 3),
        orientation=theta,
        layer=layer,
        width=width,
    )

    return c


@pp.autoname
def bend_euler90(radius=10.0, width=0.5, resolution=150.0, layer=LAYER.WG):
    """
    .. plot::
      :include-source:

      import pp

      c = pp.c.bend_euler90()
      pp.plotgds(c)

    """
    c = _bend_euler(
        theta=90, radius=radius, width=width, resolution=resolution, layer=layer
    )
    return auto_rename_ports(c)


@pp.autoname
def bend_euler90_biased(radius=10.0, width=0.5, resolution=150.0, layer=LAYER.WG):
    width = pp.bias.width(width)
    c = _bend_euler(
        theta=90, radius=radius, width=width, resolution=resolution, layer=layer
    )
    return auto_rename_ports(c)


@pp.autoname
def bend_euler180(radius=10.0, width=0.5, resolution=150.0, layer=LAYER.WG):
    """
    .. plot::
      :include-source:

      import pp

      c = pp.c.bend_euler180()
      pp.plotgds(c)

    """
    c = _bend_euler(
        theta=180, radius=radius, width=width, resolution=resolution, layer=layer
    )
    return auto_rename_ports(c)


@pp.autoname
def bend_euler180_biased(radius=10.0, width=0.5, resolution=150.0, layer=LAYER.WG):
    width = pp.bias.width(width)
    c = _bend_euler(
        theta=180, radius=radius, width=width, resolution=resolution, layer=layer
    )
    return auto_rename_ports(c)


if __name__ == "__main__":
    pass
    c = bend_euler90()
    # c = bend_euler90_biased()
    # c = bend_euler180()
    print(c.ports)
    pp.show(c)
