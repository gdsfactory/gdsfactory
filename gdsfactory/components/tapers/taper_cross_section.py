from __future__ import annotations

__all__ = [
    "taper_cross_section",
    "taper_cross_section_linear",
    "taper_cross_section_parabolic",
    "taper_cross_section_sine",
]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import CrossSectionSpec, LayerSpecs

from .._schematic import transition_schematic


@gf.cell_with_module_name(schematic_function=transition_schematic, tags=["tapers"])
def taper_cross_section(
    cross_section1: CrossSectionSpec = "strip_rib_tip",
    cross_section2: CrossSectionSpec = "rib2",
    length: float = 10,
    npoints: int = 100,
    linear: bool = False,
    width_type: str = "sine",
    exclude_layers: LayerSpecs | None = None,
) -> Component:
    r"""Returns taper transition between cross_section1 and cross_section2.

    Args:
        cross_section1: start cross_section factory.
        cross_section2: end cross_section factory.
        length: transition length.
        npoints: number of points.
        linear: shape of the transition, sine when False.
        width_type: shape of the transition ONLY IF linear is False
        exclude_layers: layers to exclude from the transition.
            Sections on these layers will be omitted from the component.

    ```text
                           _____________________
                          /
                  _______/______________________
                        /
       cross_section1  |        cross_section2
                  ______\_______________________
                         \
                          \_____________________
    ```


    """
    return cf.taper_cross_section(
        cross_section1=cross_section1,
        cross_section2=cross_section2,
        length=length,
        npoints=npoints,
        linear=linear,
        width_type=width_type,
        exclude_layers=exclude_layers,
    )


taper_cross_section_linear = CellAlias(taper_cross_section, linear=True, npoints=2)
taper_cross_section_sine = CellAlias(taper_cross_section, linear=False, npoints=101)
taper_cross_section_parabolic = CellAlias(
    taper_cross_section, linear=False, width_type="parabolic", npoints=101
)
