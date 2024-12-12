"""Greek cross test structure."""

import gdsfactory as gf
from gdsfactory.cross_section import metal1
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats, LayerSpecs


@gf.cell
def greek_cross(
    length: float = 30,
    layers: LayerSpecs = (
        "WG",
        "N",
    ),
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

    .. code::

            via_stack
            <------->
            _________       length          ________
            |       |<-------------------->|
        2x  |       |     |   ↓       |<-->|
            |       |======== width =======|
            |_______|<--> |   ↑       |<-->|________
                    offset            offset


    References:
    - Walton, Anthony J.. “MICROELECTRONIC TEST STRUCTURES.” (1999).
    - W. Versnel, Analysis of the Greek cross, a Van der Pauw structure with finite
        contacts, Solid-State Electronics, Volume 22, Issue 11, 1979, Pages 911-914,
        ISSN 0038-1101, https://doi.org/10.1016/0038-1101(79)90061-3.
    - S. Enderling et al., "Sheet resistance measurement of non-standard cleanroom
        materials using suspended Greek cross test structures," IEEE Transactions on
        Semiconductor Manufacturing, vol. 19, no. 1, pp. 2-9, Feb. 2006,
        doi: 10.1109/TSM.2005.863248.
    - https://download.tek.com/document/S530_VanDerPauwSheetRstnce.pdf

    """
    c = gf.Component()

    if len(layers) != len(widths):
        raise ValueError("len(layers) must equal len(widths).")

    offsets = offsets or (0.0,) * len(layers)
    index = 0

    for layer, width, offset in zip(layers, widths, offsets):
        ref = c << gf.c.cross(
            length=length + 2 * offset,
            width=width,
            layer=layer,
            port_type="electrical",
        )
        if index == layer_index:
            cross_ref = ref
        index += 1

    # Add via
    for port in cross_ref.ports:
        via_stack_ref = c << gf.get_component(via_stack)
        via_stack_ref.connect(
            "e1",
            port,
            allow_layer_mismatch=True,
            allow_width_mismatch=True,
        )
        c.add_port(name=port.name, port=via_stack_ref.ports["e3"])

    c.flatten()
    c.auto_rename_ports()
    return c


@gf.cell
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
    c = gf.Component()

    # Cross
    cross_ref = c << gf.get_component(greek_cross_component)
    cross_ref.dx = (
        2 * pad_pitch - (pad_pitch - gf.get_component(pad).info["size"][0]) / 2
    )

    cross_pad_via_port_pairs = {
        0: ("e1", "e2"),
        1: ("e4", "e2"),
        2: ("e2", "e4"),
        3: ("e3", "e4"),
    }

    # Vias to pads
    for index in range(4):
        pad_ref = c << gf.get_component(pad)
        pad_ref.dx = index * pad_pitch + pad_ref.dxsize / 2
        via_ref = c << gf.get_component(pad_via)
        if index < 2:
            via_ref.connect(
                "e2",
                other=pad_ref.ports["e4"],
                allow_layer_mismatch=True,
                allow_width_mismatch=True,
            )
        else:
            via_ref.connect(
                "e4",
                other=pad_ref.ports["e2"],
                allow_layer_mismatch=True,
                allow_width_mismatch=True,
            )

        gf.routing.route_single_electrical(
            c,
            cross_ref[cross_pad_via_port_pairs[index][0]],
            via_ref[cross_pad_via_port_pairs[index][1]],
            cross_section=cross_section,
            start_straight_length=5,
            end_straight_length=5,
        )
        c.add_port(
            name=f"e{index+1}",
            port=pad_ref.ports[pad_port_name],
        )

    return c


if __name__ == "__main__":
    c = greek_cross_with_pads()
    c.pprint_ports()
    c.show()
