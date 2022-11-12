import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_elliptical import grating_tooth_points
from gdsfactory.geometry.functions import DEG2RAD
from gdsfactory.types import CrossSectionSpec, LayerSpec, Optional


@gf.cell
def grating_coupler_elliptical_trenches(
    polarization: str = "te",
    fiber_marker_width: float = 11.0,
    fiber_marker_layer: Optional[LayerSpec] = "TE",
    taper_length: float = 16.6,
    taper_angle: float = 30.0,
    trenches_extra_angle: float = 9.0,
    wavelength: float = 1.53,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.343,
    neff: float = 2.638,  # tooth effective index
    ncladding: float = 1.443,  # cladding index
    layer_trench: LayerSpec = "SLAB150",
    p_start: int = 26,
    n_periods: int = 30,
    end_straight_length: float = 0.2,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Returns Grating coupler with defined trenches.

    Some foundries define the grating coupler by a shallow etch step (trenches)
    Others define the slab that they keep (see grating_coupler_elliptical)

    Args:
        polarization: 'te' or 'tm'.
        taper_length: taper length from straight I/O.
        taper_angle: grating flare angle.
        wavelength: grating transmission central wavelength.
        fiber_angle: fibre polish angle in degrees.
        grating_line_width: of the 220 ridge.
        neff: tooth effective index.
        ncladding: cladding index.
        layer_trench: for the trench.
        p_start: first tooth.
        n_periods: number of grating teeth.
        end_straight_length: at the end of straight.
        kwargs: cross_section settings.


    .. code::

                      fiber

                   /  /  /  /
                  /  /  /  /
                _|-|_|-|_|-|___
        WG  o1  ______________|

    """
    xs = gf.get_cross_section(cross_section, **kwargs)
    wg_width = xs.width
    layer = xs.layer

    # Compute some ellipse parameters
    sthc = np.sin(fiber_angle * DEG2RAD)
    d = neff**2 - ncladding**2 * sthc**2
    a1 = wavelength * neff / d
    b1 = wavelength / np.sqrt(d)
    x1 = wavelength * ncladding * sthc / d

    a1 = round(a1, 3)
    b1 = round(b1, 3)
    x1 = round(x1, 3)

    period = float(a1 + x1)
    trench_line_width = period - grating_line_width

    c = gf.Component()

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
        (xmax + end_straight_length, y),
        (xmax + end_straight_length, -y),
        (xmax, -y),
    ]
    c.add_polygon(pts, layer)

    x = np.round(taper_length + period * n_periods / 2, 3)
    if fiber_marker_layer:
        circle = gf.components.circle(
            radius=fiber_marker_width / 2, layer=fiber_marker_layer
        )
        circle_ref = c.add_ref(circle)
        circle_ref.movex(x)
    name = f"vertical_{polarization.lower()}"
    c.add_port(
        name=name,
        center=(x, 0),
        width=fiber_marker_width,
        orientation=0,
        layer=fiber_marker_layer,
        port_type=name,
    )

    c.add_port(
        name="o1",
        center=(x_output, 0),
        width=wg_width,
        orientation=180,
        layer=layer,
        cross_section=xs,
    )
    c.info["period"] = period
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength
    if xs.add_bbox:
        c = xs.add_bbox(c)
    if xs.add_pins:
        c = xs.add_pins(c)
    return c


grating_coupler_te = gf.partial(
    grating_coupler_elliptical_trenches, polarization="te", taper_angle=35
)

grating_coupler_tm = gf.partial(
    grating_coupler_elliptical_trenches,
    polarization="tm",
    neff=1.8,
    grating_line_width=0.6,
    fiber_marker_layer="TM",
)


if __name__ == "__main__":
    # c = grating_coupler_elliptical_trenches(polarization="TE")
    # print(c.polarization)
    # c = grating_coupler_te(end_straight_length=10, fiber_marker_layer=None)
    # c = grating_coupler_tm()
    # print(c.ports.keys())
    c = gf.routing.add_fiber_array(grating_coupler=grating_coupler_elliptical_trenches)
    c.show(show_ports=True)
