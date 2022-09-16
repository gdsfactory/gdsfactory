from functools import partial
from typing import Optional

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.types import ComponentSpec, CrossSectionSpec, LayerSpec

# The default values are loosely based on Taillaert et al,
#  "A Compact Two-Dimensional Grating Coupler Used
# as a Polarization Splitter", IEEE Phot. Techn. Lett. 15(9), 2003

rectangle_unit_cell = partial(
    rectangle, size=(0.3, 0.3), layer="SLAB150", centered=True, port_type=None
)


@gf.cell
def grating_coupler_dual_pol(
    unit_cell: ComponentSpec = rectangle_unit_cell,
    period_x: float = 0.58,
    period_y: float = 0.58,
    x_span: float = 11,
    y_span: float = 11,
    length_taper: float = 150.0,
    width_taper: float = 10.0,
    polarization: str = "dual",
    wavelength: float = 1.55,
    taper: ComponentSpec = taper_function,
    base_layer: Optional[LayerSpec] = "WG",
    cross_section: CrossSectionSpec = "strip",
    fiber_marker_layer: LayerSpec = "TE",
    **kwargs,
) -> Component:
    r"""2 dimensional, dual polarization grating coupler.

    Based on a photonic crystal with a unit cell that is usually an ellipse,
    a rectangle or a circle.

    Args:
        unit_cell: component describing the unit cell of the photonic crystal.
        period_x: spacing between unit cells in the x direction [um].
        period_y: spacing between unit cells in the y direction [um].
        x_span: full x span of the photonic crystal.
        y_span: full y span of the photonic crystal.
        length_taper: taper length [um].
        width_taper: width of the taper at the grating coupler side [um].
        polarization: polarizatino of the grating coupler.
        wavelength: operation wavelength [um]
        taper: function to generate the tapers.
        base_layer: layer to draw over the whole photonic crystal
            (necessary if the unit cells are etched into a base layer).
        cross_section: for the routing waveguides.
        kwargs: cross_section settings.

    .. code::

        side view
                      fiber

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___  --> unit_cells
                   base_layer |
            o1  ______________|


        top view

                   -------------
               // | o   o   o  |
        o1 __ //  | o   o   o  |
              \\  | o   o   o  |
               \\ | o   o   o  |
                   -------------
                   \\         //
                    \\       //
                         |
                         o2

    """
    xs = gf.get_cross_section(cross_section, **kwargs)
    wg_width = xs.width
    layer = xs.layer

    c = Component()

    # ---------- First draw the grating coupler itself -----

    # Base layer
    _ = c << rectangle(
        size=(x_span, y_span), layer=base_layer, centered=True, port_type=None
    )

    # Photonic crystal
    num_x = int(np.floor(x_span / period_x))
    num_y = int(np.floor(y_span / period_y))
    x_start = -(num_x * period_x) / 2
    y_start = -(num_y * period_y) / 2
    x_end = -x_start
    y_end = -y_start
    x = np.linspace(x_start, x_end, num_x)
    y = np.linspace(y_start, y_end, num_y)
    xpos, ypos = np.meshgrid(x, y)

    for x, y in zip(xpos.flatten(), ypos.flatten()):
        un_cell = c << unit_cell()
        un_cell.x = gf.snap.snap_to_grid(x)
        un_cell.y = gf.snap.snap_to_grid(y)

    port_type = f"vertical_{polarization.lower()}"
    c.add_port(
        name=port_type,
        port_type=port_type,
        center=(0, 0),
        orientation=0,
        width=x_span,
        layer=fiber_marker_layer,
    )
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength

    # --------- Now draw the tapers -----------

    # First taper
    taper1 = c << gf.get_component(
        taper,
        length=length_taper,
        width2=width_taper,
        width1=wg_width,
        layer=layer,
    )

    taper1.xmax = -x_span / 2
    taper1.y = 0
    c.add_port(port=taper1.ports["o1"], name="o1")

    # Second taper
    taper2 = c << gf.get_component(
        taper,
        length=length_taper,
        width2=width_taper,
        width1=wg_width,
        layer=layer,
    )
    taper2.rotate(90)

    taper2.x = 0
    taper2.ymax = -y_span / 2
    c.add_port(port=taper2.ports["o1"], name="o2")

    gf.asserts.grating_coupler(c)

    # -------- Some final stuff -----

    if xs.add_bbox:
        c = xs.add_bbox(c)
    if xs.add_pins:
        c = xs.add_pins(c)

    return c


if __name__ == "__main__":
    c = grating_coupler_dual_pol()
    c.show(show_ports=True)
