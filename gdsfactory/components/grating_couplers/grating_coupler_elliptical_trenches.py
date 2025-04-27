from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_couplers.functions import grating_tooth_points
from gdsfactory.functions import DEG2RAD
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec


@gf.cell_with_module_name
def grating_coupler_elliptical_trenches(
    polarization: str = "te",
    taper_length: float = 16.6,
    taper_angle: float = 30.0,
    trenches_extra_angle: float = 9.0,
    wavelength: float = 1.53,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.343,
    neff: float = 2.638,  # tooth effective index
    ncladding: float = 1.443,  # cladding index
    layer_trench: LayerSpec = "SHALLOW_ETCH",
    p_start: int = 26,
    n_periods: int = 30,
    end_straight_length: float = 0.2,
    taper: ComponentSpec = "taper",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns Grating coupler with defined trenches.

    Some foundries define the grating coupler by a shallow etch step (trenches)
    Others define the slab that they keep (see grating_coupler_elliptical)

    Args:
        polarization: 'te' or 'tm'.
        taper_length: taper length from straight I/O.
        taper_angle: grating flare angle.
        trenches_extra_angle: extra angle for the trenches.
        wavelength: grating transmission central wavelength.
        fiber_angle: fibre polish angle in degrees.
        grating_line_width: of the 220 ridge.
        neff: tooth effective index.
        ncladding: cladding index.
        layer_trench: for the trench.
        p_start: first tooth.
        n_periods: number of grating teeth.
        end_straight_length: at the end of straight.
        taper: taper function.
        cross_section: cross_section spec.


    .. code::

                      fiber

                   /  /  /  /
                  /  /  /  /
                _|-|_|-|_|-|___
        WG  o1  ______________|

    """
    xs = gf.get_cross_section(cross_section)
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

    taper_length2 = (xmax + end_straight_length) - x_output
    taper_component = c << gf.get_component(
        taper,
        width1=wg_width,
        width2=2 * y,
        length=taper_length2,
        cross_section=cross_section,
    )
    taper_component.xmin = x_output

    c.add_port(
        name="o1",
        port=taper_component.ports["o1"],
    )
    c.info["period"] = float(np.round(period, 3))
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength

    x = np.round(taper_length + period * n_periods / 2, 3)
    c.flatten()
    c.add_port(
        name="o2",
        center=(x, 0),
        width=10,
        orientation=0,
        layer=layer,
        port_type=f"vertical_{polarization}",
    )
    return c


grating_coupler_te = partial(
    grating_coupler_elliptical_trenches, polarization="te", taper_angle=35
)

grating_coupler_tm = partial(
    grating_coupler_elliptical_trenches,
    polarization="tm",
    neff=1.8,
    grating_line_width=0.6,
)


if __name__ == "__main__":
    c = grating_coupler_te(cross_section="rib_bbox")
    # c = grating_coupler_elliptical_trenches(polarization="TE")
    # print(c.polarization)
    # c = grating_coupler_te(end_straight_length=10)
    # c = grating_coupler_tm()
    # print(c.ports.keys())
    # c = gf.routing.add_fiber_array(grating_coupler=grating_coupler_elliptical_trenches)
    c.show()
