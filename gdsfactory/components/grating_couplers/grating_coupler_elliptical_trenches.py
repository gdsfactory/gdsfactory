from __future__ import annotations

__all__ = [
    "grating_coupler_elliptical_trenches",
    "grating_coupler_te",
    "grating_coupler_tm",
]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec

from .._schematic import grating_coupler_schematic


@gf.cell_with_module_name(
    schematic_function=grating_coupler_schematic, tags=["grating_couplers"]
)
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

                      fiber

    ```text
                   /  /  /  /
                  /  /  /  /
                _|-|_|-|_|-|___
        WG  o1  ______________|
    ```

    """
    return cf.grating_coupler_elliptical_trenches(
        polarization=polarization,
        taper_length=taper_length,
        taper_angle=taper_angle,
        trenches_extra_angle=trenches_extra_angle,
        wavelength=wavelength,
        fiber_angle=fiber_angle,
        grating_line_width=grating_line_width,
        neff=neff,
        ncladding=ncladding,
        layer_trench=layer_trench,
        p_start=p_start,
        n_periods=n_periods,
        end_straight_length=end_straight_length,
        taper=taper,
        cross_section=cross_section,
    )


grating_coupler_te = CellAlias(
    grating_coupler_elliptical_trenches, polarization="te", taper_angle=35
)

grating_coupler_tm = CellAlias(
    grating_coupler_elliptical_trenches,
    polarization="tm",
    neff=1.8,
    grating_line_width=0.6,
)


if __name__ == "__main__":
    c = grating_coupler_elliptical_trenches()
    s = c.to_3d()
    s.show()
