from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.types import ComponentFactory, Floats, Layer

_gaps = [0.2] * 10
_widths = [0.5] * 10


@gf.cell
def grating_coupler_rectangular_arbitrary(
    gaps: Floats = _gaps,
    widths: Floats = _widths,
    wg_width: float = 0.5,
    width_grating: float = 11.0,
    length_taper: float = 150.0,
    layer: Tuple[int, int] = gf.LAYER.WG,
    polarization: str = "te",
    wavelength: float = 1.55,
    taper: ComponentFactory = taper_function,
    layer_grating: Optional[Layer] = None,
    width_teeth: Optional[float] = None,
) -> Component:
    r"""Grating coupler uniform (grating with rectangular shape not elliptical).
    Therefore it needs a longer taper.
    Grating teeth are straight instead of elliptical.

    Args:
        gaps: list of gaps
        widths: list of widths
        wg_width:
        width_grating:
        length_taper:
        layer: for grating
        polarization:
        wavelength:
        taper: function
        layer_grating:
        width_teeth: defautls to width_grating

    .. code::

                 \  \  \  \
                  \  \  \  \
                _|-|_|-|_|-|___
               |_______________  W0

    """
    c = Component()
    taper_ref = c << taper(
        length=length_taper,
        width2=width_grating,
        width1=wg_width,
        layer=layer,
    )

    c.add_port(port=taper_ref.ports["o1"], name="o1")
    xi = taper_ref.xmax

    for width, gap in zip(widths, gaps):
        xi += gap + width / 2

        cgrating = c.add_ref(
            rectangle(
                size=[width, width_teeth or width_grating],
                layer=layer,
                port_type=None,
                centered=True,
            )
        )
        cgrating.x = gf.snap.snap_to_grid(xi)
        cgrating.y = 0
        xi += width / 2

    xport = np.round((xi + length_taper) / 2, 3)

    port_type = f"vertical_{polarization.lower()}"
    c.add_port(name=port_type, port_type=port_type, midpoint=(xport, 0), orientation=0)
    c.info.polarization = polarization
    c.info.wavelength = wavelength
    gf.asserts.grating_coupler(c)
    return c


if __name__ == "__main__":
    c = grating_coupler_rectangular_arbitrary()
    print(c.ports)
    c.show()
