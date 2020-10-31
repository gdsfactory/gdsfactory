from typing import Tuple, Union
from numpy import float64, ndarray
import numpy as np
import pp
from pp.geo_utils import extrude_path
from pp.geo_utils import DEG2RAD
from pp.layers import LAYER
from pp.component import Component


def ellipse_arc(
    a: float64,
    b: float64,
    x0: float64,
    theta_min: float,
    theta_max: float,
    angle_step: float = 0.5,
) -> ndarray:
    theta = np.arange(theta_min, theta_max + angle_step, angle_step) * DEG2RAD
    xs = a * np.cos(theta) + x0
    ys = b * np.sin(theta)
    return np.column_stack([xs, ys])


def grating_tooth_points(
    ap: float64,
    bp: float64,
    xp: float64,
    width: Union[float64, float],
    taper_angle: float,
    spiked: bool = True,
    angle_step: float = 1.0,
) -> ndarray:
    theta_min = -taper_angle / 2
    theta_max = taper_angle / 2

    backbone_points = ellipse_arc(ap, bp, xp, theta_min, theta_max, angle_step)
    if spiked:
        spike_length = width / 3
    else:
        spike_length = 0.0
    points = extrude_path(
        backbone_points,
        width,
        with_manhattan_facing_angles=False,
        spike_length=spike_length,
    )

    return points


def grating_taper_points(
    a: float64,
    b: float64,
    x0: float64,
    taper_length: float64,
    taper_angle: float,
    wg_width: float,
    angle_step: float = 1.0,
) -> ndarray:
    taper_arc = ellipse_arc(a, b, taper_length, -taper_angle / 2, taper_angle / 2)

    port_position = np.array((x0, 0))
    p0 = port_position + (0, wg_width / 2)
    p1 = port_position + (0, -wg_width / 2)
    points = np.vstack([p0, p1, taper_arc])
    return points


@pp.autoname
def grating_coupler_elliptical_tm(
    taper_length: float = 16.6,
    taper_angle: float = 30.0,
    lambda_c: float = 1.554,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.707,
    wg_width: float = 0.5,
    neff: float = 1.8,  # tooth effective index
    layer: Tuple[int, int] = LAYER.WG,
    n_periods: int = 16,
    **kwargs
) -> Component:
    """

    Args:
        neff: tooth effective index

    .. plot::
      :include-source:

      import pp

      c = pp.c.grating_coupler_elliptical_tm()
      pp.plotgds(c)

    """
    return grating_coupler_elliptical(
        polarization="tm",
        taper_length=taper_length,
        taper_angle=taper_angle,
        lambda_c=lambda_c,
        fiber_angle=fiber_angle,
        grating_line_width=grating_line_width,
        wg_width=wg_width,
        neff=neff,
        layer=layer,
        n_periods=n_periods,
        big_last_tooth=False,
        **kwargs
    )


@pp.autoname
def grating_coupler_elliptical_te(
    taper_length: float = 16.6,
    taper_angle: float = 40.0,
    lambda_c: float = 1.554,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.343,
    wg_width: float = 0.5,
    neff: float = 2.638,  # tooth effective index
    layer: Tuple[int, int] = LAYER.WG,
    p_start: int = 26,
    n_periods: int = 24,
    **kwargs
) -> Component:
    return grating_coupler_elliptical(
        polarization="te",
        taper_length=taper_length,
        taper_angle=taper_angle,
        lambda_c=lambda_c,
        fiber_angle=fiber_angle,
        grating_line_width=grating_line_width,
        wg_width=wg_width,
        neff=neff,
        layer=layer,
        p_start=p_start,
        n_periods=n_periods,
        **kwargs
    )


@pp.autoname
def grating_coupler_elliptical(
    polarization: str,
    taper_length: float = 16.6,
    taper_angle: float = 30.0,
    lambda_c: float = 1.554,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.343,
    wg_width: float = 0.5,
    neff: float = 2.638,  # tooth effective index
    layer: Tuple[int, int] = LAYER.WG,
    p_start: int = 26,
    n_periods: int = 30,
    big_last_tooth: bool = False,
    layer_slab: Tuple[int, int] = LAYER.SLAB150,
    with_fiber_marker: bool = True,
) -> Component:
    """

    Args:
        taper_length: taper length from waveguide I/O
        taper_angle: grating flare angle
        lambda_c: grating transmission central wavelength (um)
        fiber_angle: fibre polish angle in degrees
        grating_line_width
        wg_width: waveguide width
        neff: 2.638  # tooth effective index
        layer: LAYER.WG
        p_start: 26  # first tooth
        n_periods: 16  # number of periods

    .. plot::
      :include-source:

      import pp

      c = pp.c.grating_coupler_elliptical_te()
      pp.plotgds(c)
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

    c = pp.Component()
    c.polarization = polarization
    c.wavelength = int(lambda_c * 1e3)

    # Make each grating line
    for p in range(p_start, p_start + n_periods + 1):
        pts = grating_tooth_points(
            p * a1, p * b1, p * x1, grating_line_width, taper_angle
        )
        c.add_polygon(pts, layer)

    # Make the taper
    p_taper = p_start - 1
    p_taper_eff = p_taper
    a_taper = a1 * p_taper_eff
    b_taper = b1 * p_taper_eff
    x_taper = x1 * p_taper_eff

    x_output = a_taper + x_taper - taper_length + grating_line_width / 2
    pts = grating_taper_points(
        a_taper, b_taper, x_output, x_taper, taper_angle, wg_width=wg_width
    )
    c.add_polygon(pts, layer)

    # Superimpose a tooth without spikes at end of taper to match the period.
    pts = grating_tooth_points(
        a_taper, b_taper, x_taper, grating_line_width, taper_angle, spiked=False
    )
    c.add_polygon(pts, layer)

    # Add last "large tooth" after the standard grating teeth
    w = 1.0
    L = (
        period * (p_start + n_periods)
        + grating_line_width / 2
        + period
        - grating_line_width
        + w / 2
    )

    if big_last_tooth:
        a = L / (1 + x1 / a1)
        b = b1 / a1 * a
        x = x1 / a1 * a

        pts = grating_tooth_points(a, b, x, w, taper_angle, spiked=False)
        c.add_polygon(pts, layer)

    # Move waveguide I/O to (0, 0)
    c.move((-x_output, 0))

    if polarization.lower() == "te":
        polarization_marker_layer = pp.LAYER.TE
    else:
        polarization_marker_layer = pp.LAYER.TM

    if with_fiber_marker:
        circle = pp.c.circle(radius=17 / 2, layer=polarization_marker_layer)
        circle_ref = c.add_ref(circle)
        circle_ref.movex(taper_length + period * n_periods / 2)

    # Add port
    c.add_port(name="W0", midpoint=[0, 0], width=wg_width, orientation=180, layer=layer)

    # Add shallow etch
    _rl = L + grating_line_width + 2.0
    _rhw = _rl * np.tan(fiber_angle * DEG2RAD) + 2.0

    if layer_slab:
        c.add_polygon([(0, _rhw), (_rl, _rhw), (_rl, -_rhw), (0, -_rhw)], LAYER.SLAB150)

    return c


if __name__ == "__main__":
    c = grating_coupler_elliptical_tm()
    c = grating_coupler_elliptical_te(layer_slab=None, with_fiber_marker=False)
    print(c.polarization)
    print(c.wavelength)
    pp.write_gds(c)
    pp.show(c)
