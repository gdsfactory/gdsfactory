"""Greek cross test structure."""

__all__ = ["greek_cross", "greek_cross_with_pads"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.cross_section import metal1
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats, LayerSpecs


@gf.cell_with_module_name(tags=["pcms"])
def greek_cross(
    length: float = 30,
    layers: LayerSpecs = ("WG", "N"),
    widths: Floats = (2.0, 3.0),
    offsets: Floats | None = None,
    via_stack: ComponentSpec = "via_stack_npp_m1",
    layer_index: int = 0,
) -> gf.Component:
    """Simple greek cross with via stacks at the endpoints.

    Process control monitor for dopant sheet resistivity and linewidth variation.

    Args:
        length: length of cross arms.
        layers: list of layers.
        widths: list of widths (same order as layers).
        offsets: how much to extend each layer beyond the cross length
            negative shorter, positive longer.
        via_stack: via component to attach to the cross.
        layer_index: index of the layer to connect the via_stack to.

    ```text
            via_stack
            <------->
            _________       length          ________
            |       |<-------------------->|        |
        2x  |       |     |   ↓       |<-->|        |
            |       |======== width =======|        |
            |_______|<--> |   ↑       |<-->|________|
                    offset            offset
    ```


    References:
    - Walton, Anthony J.. “MICROELECTRONIC TEST STRUCTURES.” (1999).
    - W. Versnel, Analysis of the Greek cross, a Van der Pauw structure with finite
      contacts, Solid-State Electronics, Volume 22, Issue 11, 1979, Pages 911-914,
      ISSN 0038-1101, <https://doi.org/10.1016/0038-1101>(79)90061-3.
    - S. Enderling et al., "Sheet resistance measurement of non-standard cleanroom
      materials using suspended Greek cross test structures," IEEE Transactions on
      Semiconductor Manufacturing, vol. 19, no. 1, pp. 2-9, Feb. 2006,
      doi: 10.1109/TSM.2005.863248.
    - <https://download.tek.com/document/S530_VanDerPauwSheetRstnce.pdf>

    """
    return cf.greek_cross(
        length=length,
        layers=layers,
        widths=widths,
        offsets=offsets,
        via_stack=via_stack,
        layer_index=layer_index,
    )


@gf.cell_with_module_name(tags=["pcms"])
def greek_cross_with_pads(
    pad: ComponentSpec = "pad",
    pad_pitch: float = 150.0,
    greek_cross_component: ComponentSpec = "greek_cross",
    pad_via: ComponentSpec = "via_stack_m1_mtop",
    cross_section: CrossSectionSpec = metal1,
    pad_port_name: str = "e4",
) -> gf.Component:
    """Greek cross under 4 DC pads, ready to test.

    Arguments:
        pad: component to use for probe pads.
        pad_pitch: spacing between pads.
        greek_cross_component: component to use for greek cross.
        pad_via: via to add to the pad.
        cross_section: cross-section for cross via to pad via wiring.
        pad_port_name: name of the port to connect to the greek cross.
    """
    return cf.greek_cross_with_pads(
        pad=pad,
        pad_pitch=pad_pitch,
        greek_cross_component=greek_cross_component,
        pad_via=pad_via,
        cross_section=cross_section,
        pad_port_name=pad_port_name,
    )
