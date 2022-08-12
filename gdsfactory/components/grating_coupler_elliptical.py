from typing import Optional

import numpy as np
from numpy import ndarray

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.geometry.functions import DEG2RAD, extrude_path
from gdsfactory.types import CrossSectionSpec, LayerSpec


def ellipse_arc(
    a: float,
    b: float,
    x0: float,
    theta_min: float,
    theta_max: float,
    angle_step: float = 0.5,
) -> ndarray:
    """Returns an elliptical arc.

    b = a *sqrt(1-e**2)

    An ellipse with a = b has zero eccentricity (is a circle)

    Args:
        a: ellipse semi-major axis.
        b: semi-minor axis.
        x0: in um.
        theta_min: in rad.
        theta_max: in rad.
        angle_step: in rad.
    """
    theta = np.arange(theta_min, theta_max + angle_step, angle_step) * DEG2RAD
    xs = a * np.cos(theta) + x0
    ys = b * np.sin(theta)
    return np.column_stack([xs, ys])


def grating_tooth_points(
    ap: float,
    bp: float,
    xp: float,
    width: float,
    taper_angle: float,
    spiked: bool = True,
    angle_step: float = 1.0,
) -> ndarray:
    theta_min = -taper_angle / 2
    theta_max = taper_angle / 2

    backbone_points = ellipse_arc(ap, bp, xp, theta_min, theta_max, angle_step)
    spike_length = width / 3 if spiked else 0.0
    return extrude_path(
        backbone_points,
        width,
        with_manhattan_facing_angles=False,
        spike_length=spike_length,
    )


def grating_taper_points(
    a: float,
    b: float,
    x0: float,
    taper_length: float,
    taper_angle: float,
    wg_width: float,
    angle_step: float = 1.0,
) -> ndarray:
    taper_arc = ellipse_arc(a, b, taper_length, -taper_angle / 2, taper_angle / 2)

    port_position = np.array((x0, 0))
    p0 = port_position + (0, wg_width / 2)
    p1 = port_position + (0, -wg_width / 2)
    return np.vstack([p0, p1, taper_arc])


@gf.cell
def grating_coupler_elliptical(
    polarization: str = "te",
    taper_length: float = 16.6,
    taper_angle: float = 40.0,
    wavelength: float = 1.554,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.343,
    neff: float = 2.638,  # tooth effective index
    nclad: float = 1.443,
    n_periods: int = 30,
    big_last_tooth: bool = False,
    layer_slab: LayerSpec = "SLAB150",
    slab_xmin: float = -1.0,
    slab_offset: float = 2.0,
    fiber_marker_width: Optional[float] = 11.0,
    fiber_marker_layer: LayerSpec = "TE",
    spiked: bool = True,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Grating coupler with parametrization based on Lumerical FDTD simulation.

    Args:
        polarization: te or tm.
        taper_length: taper length from input.
        taper_angle: grating flare angle.
        wavelength: grating transmission central wavelength (um).
        fiber_angle: fibre angle in degrees determines ellipticity.
        grating_line_width: in um.
        neff: tooth effective index.
        nclad: cladding effective index.
        n_periods: number of periods.
        big_last_tooth: adds a big_last_tooth.
        layer_slab: layer that protects the slab under the grating.
        slab_xmin: where 0 is at the start of the taper.
        slab_offset: in um.
        fiber_marker_width: width in um.
        fiber_marker_layer: fiber marker layer.
        spiked: grating teeth have sharp spikes to avoid non-manhattan drc errors.
        cross_section: specification (CrossSection, string or dict).
        kwargs: cross_section settings.

    .. code::

                      fiber

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|

    """
    xs = gf.get_cross_section(cross_section, **kwargs)

    wg_width = xs.width
    layer = xs.layer

    # Compute some ellipse parameters
    sthc = np.sin(fiber_angle * DEG2RAD)
    d = neff**2 - nclad**2 * sthc**2
    a1 = wavelength * neff / d
    b1 = wavelength / np.sqrt(d)
    x1 = wavelength * nclad * sthc / d

    a1 = round(a1, 3)
    b1 = round(b1, 3)
    x1 = round(x1, 3)

    period = a1 + x1

    c = gf.Component()
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength

    # Make the taper
    p = taper_length / period
    a_taper = a1 * p
    b_taper = b1 * p
    x_taper = x1 * p

    x_output = a_taper + x_taper - taper_length
    pts = grating_taper_points(
        a=a_taper,
        b=b_taper,
        x0=x_output,
        taper_length=x_taper,
        taper_angle=taper_angle,
        wg_width=wg_width,
    )
    c.add_polygon(pts, layer)

    width = gf.snap.snap_to_grid(grating_line_width)
    gap = gf.snap.snap_to_grid(period - grating_line_width)

    xi = taper_length
    for p in range(n_periods):
        xi += gap + width / 2
        p = xi / period
        pts = grating_tooth_points(
            p * a1, p * b1, p * x1, width, taper_angle, spiked=spiked
        )
        c.add_polygon(pts, layer)
        xi += width / 2

    w = 1.0
    total_length = (
        period * n_periods
        + taper_length
        + grating_line_width / 2
        + period
        - grating_line_width
        + w / 2
    )

    if big_last_tooth:
        # Add last "large tooth" after the standard grating teeth

        a = total_length / (1 + x1 / a1)
        b = b1 / a1 * a
        x = x1 / a1 * a

        pts = grating_tooth_points(a, b, x, w, taper_angle, spiked=False)
        c.add_polygon(pts, layer)

    x = np.round(taper_length + x_output, 3)
    if fiber_marker_width:
        circle = gf.components.circle(
            radius=fiber_marker_width / 2, layer=fiber_marker_layer
        )
        circle_ref = c.add_ref(circle)
        circle_ref.movex(x + fiber_marker_width / 2)

    name = f"vertical_{polarization.lower()}"
    c.add_port(
        name=name,
        center=(x + fiber_marker_width / 2, 0),
        width=fiber_marker_width,
        orientation=0,
        layer=fiber_marker_layer,
        port_type=name,
    )

    c.add_port(
        name="o1", center=(x_output, 0), width=wg_width, orientation=180, layer=layer
    )

    if layer_slab:
        slab_xmin += x_output + taper_length
        slab_length = total_length + slab_offset
        slab_width = (c.ysize + 2 * slab_offset) / 2
        c.add_polygon(
            [
                (slab_xmin, slab_width),
                (slab_length, slab_width),
                (slab_length, -slab_width),
                (slab_xmin, -slab_width),
            ],
            layer_slab,
        )

    if xs.add_bbox:
        c = xs.add_bbox(c)
    if xs.add_pins:
        c = xs.add_pins(c)
    return c


grating_coupler_elliptical_tm = gf.partial(
    grating_coupler_elliptical,
    grating_line_width=0.707,
    fiber_marker_layer="TM",
    polarization="tm",
    taper_length=30,
    slab_xmin=-2,
    neff=1.8,
    n_periods=16,
)


grating_coupler_elliptical_te = grating_coupler_elliptical


if __name__ == "__main__":
    # c = grating_coupler_elliptical_tm(taper_length=30)
    c = grating_coupler_elliptical_te(cladding_layers=((2, 0), (3, 0)))
    # c = grating_coupler_elliptical(layer=(2, 0), taper_length=50, slab_xmin=-5)
    # print(c.polarization)
    # print(c.wavelength)
    # print(c.ports)
    # c.pprint()
    # c = gf.c.extend_ports(c)
    # c = gf.routing.add_fiber_array(grating_coupler=grating_coupler_elliptical, with_loopback=False)
    c.show(show_ports=True)
