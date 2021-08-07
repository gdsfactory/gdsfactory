from typing import Tuple, Union

import gdspy as gds
import numpy as np
from gdspy.polygon import Polygon
from numpy import float64

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.snap import snap_to_grid


def taper(
    start_width: Union[float, float64],
    end_width: Union[float, float64],
    length: Union[float, float64],
    start_coord: Tuple[float64, float64],
    layer: int = gf.LAYER.WG,
    datatype: int = 0,
) -> Tuple[Polygon, Tuple[float64, float64], Tuple[float64, float64]]:
    s = start_coord
    top_left = (s[0], s[1] + start_width / 2.0)
    bot_left = (s[0], s[1] - start_width / 2.0)
    top_right = (s[0] + length, s[1] + end_width / 2.0)
    bot_right = (s[0] + length, s[1] - end_width / 2.0)
    e = (s[0] + length, s[1])
    p = gds.Polygon(
        [top_left, bot_left, bot_right, top_right], layer=layer, datatype=datatype
    )
    return p, s, e


def straight(
    width: Union[float, float64],
    length: Union[float, float64],
    start_coord: Tuple[float64, float64],
    layer: int = gf.LAYER.WG,
    datatype: int = 0,
) -> Tuple[Polygon, Tuple[float64, float64], Tuple[float64, float64]]:
    t, s, e = taper(width, width, length, start_coord, layer=layer, datatype=datatype)
    return t, s, e


@gf.cell
def spiral_circular(
    length: float = 1e3,
    wg_width: float = 0.5,
    spacing: float = 3.0,
    min_bend_radius: float = 5.0,
    points: int = 1000,
    layer: Tuple[int, int] = gf.LAYER.WG,
) -> Component:
    """Returns a circular spiral.

    Args:
        length: length in um
        wg_width:
        spacing: between straights
        min_bend_radius:
        points:
        layer:

    """
    wg_datatype = layer[1]
    wg_layer = layer[0]

    def pol_to_rect(radii, angles_deg):
        angles_rad = np.radians(angles_deg)
        z = radii * np.exp(1.0j * angles_rad)
        return z.real, z.imag

    ps = []

    # Estimate number of revolutions
    length_total = 0.0
    i = 0
    while length_total <= length:
        length_total += 3.0 * np.pi * (min_bend_radius * 2.0 + (i + 0.5) * spacing)
        i += 1
    revolutions = i + 1

    # Long spiral
    inner_revs = min_bend_radius * 4.0 / (spacing * 4.0)
    theta_1 = np.linspace(360.0 * inner_revs, 360.0 * (revolutions - 1) + 270, points)
    theta_2 = np.linspace(360.0 * inner_revs, 360.0 * revolutions, points)
    a = np.sqrt(spacing / 180.0)
    radii_1 = a ** 2 * theta_1
    radii_2 = -(a ** 2) * theta_2
    x_1, y_1 = pol_to_rect(radii_1, theta_1)
    x_1 = np.append(x_1, x_1[-1] + 0.03)
    y_1 = np.append(y_1, y_1[-1])

    p = gds.PolyPath(np.c_[x_1, y_1], wg_width, layer=wg_layer, datatype=wg_datatype)
    ps.append(p)
    x_2, y_2 = pol_to_rect(radii_2, theta_2)
    x_2 = np.append(x_2, x_2[-1])
    y_2 = np.append(y_2, y_2[-1] - 0.03)
    p = gds.PolyPath(np.c_[x_2, y_2], wg_width, layer=wg_layer, datatype=wg_datatype)
    ps.append(p)

    start_1 = (x_1[-1], y_1[-1])
    start_2 = (x_2[-1], y_2[-1])
    end_1 = (x_1[0], y_1[0])
    end_2 = (x_2[0], y_2[0])

    length_1 = np.sum(np.hypot(np.diff(x_1), np.diff(y_1)))
    length_2 = np.sum(np.hypot(np.diff(x_2), np.diff(y_2)))

    # Inner bend
    theta_inner = np.linspace(360.0 * inner_revs, 360.0 * inner_revs + 180.0, 50)
    radii_inner_1 = min_bend_radius
    radii_inner_2 = -min_bend_radius
    x_1, y_1 = pol_to_rect(radii_inner_1, theta_inner)
    x_1 -= end_1[0] / 2.0
    y_1 -= end_1[1] / 2.0
    x_1 = np.append(x_1, x_1[-1])
    y_1 = np.append(y_1, y_1[-1] - 0.03)
    p = gds.PolyPath(np.c_[x_1, y_1], wg_width, layer=wg_layer, datatype=wg_datatype)
    ps.append(p)
    x_2, y_2 = pol_to_rect(radii_inner_2, theta_inner)
    x_2 -= end_2[0] / 2.0
    y_2 -= end_2[1] / 2.0
    x_2 = np.append(x_2, x_2[-1])
    y_2 = np.append(y_2, y_2[-1] + 0.03)
    p = gds.PolyPath(np.c_[x_2, y_2], wg_width, layer=wg_layer, datatype=wg_datatype)
    ps.append(p)
    length_3 = np.sum(np.hypot(np.diff(x_2), np.diff(y_2)))

    # Output straight
    p, _, e = straight(
        wg_width, radii_1[-1], start_1, layer=wg_layer, datatype=wg_datatype
    )
    ps.append(p)

    # Outer bend
    theta_input = np.linspace(0.0, -90.0, 50)
    r_input = min_bend_radius * 2.0
    x_input, y_input = pol_to_rect(r_input, theta_input)
    x_input += start_2[0] - r_input
    y_input += start_1[1] + r_input
    s = (x_input[-1], y_input[-1])
    end_input = (x_input[0], y_input[0])
    x_input = np.append(x_input[0], x_input)
    y_input = np.append(y_input[0] + 0.03, y_input)
    x_input = np.append(x_input, x_input[-1] - 0.03)
    y_input = np.append(y_input, y_input[-1])
    p = gds.PolyPath(
        np.c_[x_input, y_input], wg_width, layer=wg_layer, datatype=wg_datatype
    )
    ps.append(p)
    length = start_2[1] - end_input[1]
    p, _, _ = straight(
        length,
        wg_width,
        (end_input[0] - wg_width / 2.0, end_input[1] + length / 2.0),
        layer=wg_layer,
        datatype=wg_datatype,
    )
    ps.append(p)

    length = length_1 + length_2 + length_3

    # Electrode ring
    # inner_radius = min_bend_radius * 2.
    # outer_radius = a**2 * theta_2[-1]
    # mid_radius = 0.5*(inner_radius + outer_radius)
    # thickness = outer_radius - mid_radius
    # r = gds.Round((0.,0.), outer_radius, inner_radius, layer=1, max_points=1000)

    ps = gds.fast_boolean(ps, None, "or")

    c = gf.Component()
    c.length = snap_to_grid(length)
    c.add_polygon(ps, layer=layer)

    c.add_port(
        name="W0",
        midpoint=(s[0], s[1]),
        orientation=180,
        layer=layer,
        width=wg_width,
    )
    c.add_port(
        name="E0",
        midpoint=(e[0], e[1]),
        orientation=180,
        layer=layer,
        width=wg_width,
    )
    return c


if __name__ == "__main__":
    c = spiral_circular(length=1e3)
    print(c.ports)
    print(c.ports.keys())
    print(c.get_ports_array())
    c.show()
    c.write_gds()
