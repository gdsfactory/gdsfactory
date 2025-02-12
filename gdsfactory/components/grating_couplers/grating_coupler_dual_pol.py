from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec


def _unit_cell() -> gf.Component:
    return gf.components.rectangle(
        size=(0.3, 0.3), layer="SLAB150", centered=True, port_type=None
    )


@gf.cell
def grating_coupler_dual_pol(
    unit_cell: ComponentSpec = _unit_cell,
    period_x: float = 0.58,
    period_y: float = 0.58,
    x_span: float = 11,
    y_span: float = 11,
    length_taper: float = 150.0,
    width_taper: float = 10.0,
    polarization: str = "te",
    wavelength: float = 1.55,
    taper: ComponentSpec = "taper",
    base_layer: LayerSpec = "WG",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""2 dimensional, dual polarization grating coupler.

    Based on a photonic crystal with a unit cell that is usually an ellipse,
    a rectangle or a circle.
    # The default values are loosely based on Taillaert et al,
    # "A Compact Two-Dimensional Grating Coupler Used as a Polarization Splitter", IEEE Phot. Techn. Lett. 15(9), 2003

    Args:
        unit_cell: component describing the unit cell of the photonic crystal.
        period_x: spacing between unit cells in the x direction [um].
        period_y: spacing between unit cells in the y direction [um].
        x_span: full x span of the photonic crystal.
        y_span: full y span of the photonic crystal.
        length_taper: taper length [um].
        width_taper: width of the taper at the grating coupler side [um].
        polarization: polarization of the grating coupler.
        wavelength: operation wavelength [um]
        taper: function to generate the tapers.
        base_layer: layer to draw over the whole photonic crystal \
                (necessary if the unit cells are etched into a base layer).
        cross_section: for the routing waveguides.

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
    xs = gf.get_cross_section(cross_section)
    wg_width = xs.width
    layer = xs.layer

    c = Component()

    _ = c << gf.c.rectangle(
        size=(x_span, y_span), layer=base_layer, centered=True, port_type=None
    )

    # Photonic crystal
    num_x = int(np.floor(x_span / period_x))
    num_y = int(np.floor(y_span / period_y))
    x_start = -(num_x * period_x) / 2
    y_start = -(num_y * period_y) / 2

    unit_cell_grating = gf.get_component(unit_cell)
    g = c.add_ref(
        unit_cell_grating,
        columns=num_x,
        rows=num_y,
        column_pitch=period_x,
        row_pitch=period_y,
    )
    g.dxmin = x_start
    g.dymin = y_start

    port_type = f"vertical_{polarization.lower()}"
    c.add_port(
        name=port_type,
        port_type=port_type,
        center=(0, 0),
        orientation=0,
        width=x_span,
        layer=layer,
    )
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength
    taper = gf.get_component(
        taper,
        length=length_taper,
        width2=width_taper,
        width1=wg_width,
        cross_section=cross_section,
    )

    taper1 = c << taper
    taper1.dxmax = -x_span / 2
    taper1.dy = 0
    c.add_port(port=taper1.ports["o1"], name="o1")

    taper2 = c << taper
    taper2.drotate(90)
    taper2.dx = 0
    taper2.dymax = -y_span / 2
    c.add_port(port=taper2.ports["o1"], name="o2")

    xs.add_bbox(c)
    return c


if __name__ == "__main__":
    c = grating_coupler_dual_pol()
    c.show()
