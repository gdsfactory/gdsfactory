from __future__ import annotations

__all__ = ["grating_coupler_dual_pol", "grating_coupler_dual_pol_unit_cell"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec

from .._schematic import grating_coupler_schematic
from ..shapes.rectangle import rectangle

grating_coupler_dual_pol_unit_cell = CellAlias(
    rectangle, size=(0.3, 0.3), layer="SLAB150", centered=True, port_type=None
)


@gf.cell_with_module_name(
    schematic_function=grating_coupler_schematic, tags=["grating_couplers"]
)
def grating_coupler_dual_pol(
    unit_cell: ComponentSpec = "grating_coupler_dual_pol_unit_cell",
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
    The default values are loosely based on Taillaert et al,
    "A Compact Two-Dimensional Grating Coupler Used as a Polarization Splitter",
    IEEE Phot. Techn. Lett. 15(9), 2003.

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
        base_layer: layer to draw over the whole photonic crystal
            (necessary if the unit cells are etched into a base layer).
        cross_section: for the routing waveguides.

        side view
                      fiber

                   /  /  /  /
                  /  /  /  /

    ```text
                _|-|_|-|_|-|___  --> unit_cells
                   base_layer |
            o1  ______________|
    ```


        top view

    ```text
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
    ```

    """
    return cf.grating_coupler_dual_pol(
        unit_cell=unit_cell,
        period_x=period_x,
        period_y=period_y,
        x_span=x_span,
        y_span=y_span,
        length_taper=length_taper,
        width_taper=width_taper,
        polarization=polarization,
        wavelength=wavelength,
        taper=taper,
        base_layer=base_layer,
        cross_section=cross_section,
    )
