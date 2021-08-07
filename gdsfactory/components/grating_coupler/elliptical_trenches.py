from typing import Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler.elliptical import grating_tooth_points
from gdsfactory.geo_utils import DEG2RAD
from gdsfactory.types import Layer


@gf.cell
def grating_coupler_elliptical_trenches(
    polarization: str = "te",
    fiber_marker_width: float = 11.0,
    fiber_marker_layer: Layer = gf.LAYER.TE,
    taper_length: float = 16.6,
    taper_angle: float = 30.0,
    trenches_extra_angle: float = 9.0,
    lambda_c: float = 1.53,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.343,
    wg_width: float = 0.5,
    neff: float = 2.638,  # tooth effective index
    layer: Tuple[int, int] = gf.LAYER.WG,
    layer_trench: Tuple[int, int] = gf.LAYER.SLAB150,
    p_start: int = 26,
    n_periods: int = 30,
    straight: float = 0.2,
) -> Component:
    r""" Returns Grating coupler with defined trenches

    Args:
        polarization: 'te' or 'tm'
        taper_length: taper length from straight I/O
        taper_angle: grating flare angle
        lambda_c: grating transmission central wavelength
        fiber_angle: fibre polish angle in degrees
        grating_line_width: of the 220 ridge
        wg_width: straight width
        neff: 2.638  # tooth effective index
        layer: LAYER.WG
        layer_trench: LAYER.SLAB150
        p_start: 26  # first tooth
        n_periods: 16  # number of periods
        straight: 0.2

    .. plot::
      :include-source:

      import gdsfactory as gf
      from gdsfactory.components.grating_coupler.elliptical_trenches import grating_coupler_elliptical_trenches

      c = grating_coupler_elliptical_trenches()
      c.plot()


    .. code::

                 \  \  \  \
                  \  \  \  \
                _|-|_|-|_|-|___
               |_______________  W0
    """

    # Define some constants
    nc = 1.443  # cladding index

    # Compute some ellipse parameters
    sthc = np.sin(fiber_angle * DEG2RAD)
    d = neff ** 2 - nc ** 2 * sthc ** 2
    a1 = lambda_c * neff / d
    b1 = lambda_c / np.sqrt(d)
    x1 = lambda_c * nc * sthc / d

    a1 = round(a1, 3)
    b1 = round(b1, 3)
    x1 = round(x1, 3)

    period = a1 + x1
    trench_line_width = period - grating_line_width

    c = gf.Component()
    c.polarization = polarization
    c.wavelength = int(lambda_c * 1e3)

    # Make each grating line
    for p in range(p_start, p_start + n_periods + 1):
        pts = grating_tooth_points(
            p * a1,
            p * b1,
            p * x1,
            width=trench_line_width,
            taper_angle=taper_angle + trenches_extra_angle,
        )
        c.add_polygon(pts, layer_trench)

    # Make the taper
    p_taper = p_start - 1
    p_taper_eff = p_taper
    a_taper = a1 * p_taper_eff
    # b_taper = b1 * p_taper_eff
    x_taper = x1 * p_taper_eff
    x_output = a_taper + x_taper - taper_length + grating_line_width / 2

    xmax = x_output + taper_length + n_periods * period + 3
    y = wg_width / 2 + np.tan(taper_angle / 2 * np.pi / 180) * xmax
    pts = [
        (x_output, -wg_width / 2),
        (x_output, wg_width / 2),
        (xmax, y),
        (xmax + straight, y),
        (xmax + straight, -y),
        (xmax, -y),
    ]
    c.add_polygon(pts, layer)

    # a_taper = (p_start + n_periods + 1)*a1
    # b_taper = (p_start + n_periods + 1)*b1
    # pts = grating_taper_points(
    #     a_taper, b_taper, x_output, x_taper, taper_angle, wg_width=wg_width
    # )
    # c.add_polygon(pts, layer)

    # Move straight I/O to (0, 0)
    c.move((-x_output, 0))

    x = taper_length + period * n_periods / 2
    circle = gf.components.circle(
        radius=fiber_marker_width / 2, layer=fiber_marker_layer
    )
    circle_ref = c.add_ref(circle)
    circle_ref.movex(x)
    c.add_port(
        name=f"vertical_{polarization.lower()}",
        midpoint=[x, 0],
        width=fiber_marker_width,
        orientation=0,
        layer=fiber_marker_layer,
        port_type=f"vertical_{polarization.lower()}",
    )

    # Add port
    c.add_port(name="W0", midpoint=[0, 0], width=wg_width, orientation=180, layer=layer)
    c.settings["period"] = period
    return c


def grating_coupler_te(taper_angle: int = 35, **kwargs) -> Component:
    """

    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.grating_coupler_te()
      c.plot()
    """
    return grating_coupler_elliptical_trenches(
        polarization="te", taper_angle=taper_angle, **kwargs
    )


def grating_coupler_tm(
    neff: float = 1.8, grating_line_width: float = 0.6, **kwargs
) -> Component:
    """

    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.grating_coupler_tm()
      c.plot()
    """
    return grating_coupler_elliptical_trenches(
        polarization="tm", neff=neff, grating_line_width=grating_line_width, **kwargs
    )


if __name__ == "__main__":
    # c = grating_coupler_elliptical_trenches(polarization="TE")
    # print(c.polarization)
    c = grating_coupler_te()
    # c = grating_coupler_tm()
    print(c.ports.keys())
    c.show()
