"""Greek cross test structure."""
from itertools import product

import numpy as np

import gdsfactory as gf
from gdsfactory.components.cross import cross
from gdsfactory.components.pad import pad
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.via_stack import via_stack, via_stack_m1_m3, via_stack_npp_m1
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
    via_stack: ComponentSpec = via_stack_npp_m1,
) -> gf.Component:
    """Simple greek cross with via stacks at the endpoints.

    Process control monitor for dopant sheet resistivity and linewidth variation.

    Args:
        length: length of cross arms.
        layers: list of layers.
        widths: list of widths (same order as layers).
        offsets: how much to extend each layer beyond the cross length
            negative shorter, positive longer.
        via: via component to attach to the cross.

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

    # Layout cross
    for layer, width, offset in zip(layers, widths, offsets):
        cross_ref = c << gf.get_component(
            cross,
            length=length + 2 * offset,
            width=width,
            layer=layer,
            port_type="electrical",
        )
        cross_offset = offset

    port_at_length = [
        port.move_polar_copy(d=cross_offset, angle=180 + port.orientation)
        for port in cross_ref.get_ports_list()
    ]

    # Add via
    for port in port_at_length:
        via_stack_ref = c << gf.get_component(via_stack)
        via_stack_ref.connect("e1", port)
        c.add_port(name=port.name, port=via_stack_ref.ports["e3"])

    c.auto_rename_ports()

    return c


@gf.cell
def greek_cross_with_pads(
    pad: ComponentSpec = pad,
    pad_spacing: float = 150.0,
    greek_cross_component: ComponentSpec = greek_cross,
    pad_via: ComponentSpec = via_stack_m1_m3,
    xs_metal: CrossSectionSpec = metal1,
) -> gf.Component:
    """Greek cross under 4 DC pads, ready to test.

    Arguments:
        pad: component to use for probe pads
        pad_spacing: spacing between pads
        greek_cross_component: component to use for greek cross
        pad_via: via to add to the pad
        xs_metal: cross-section for cross via to pad via wiring
    """
    c = gf.Component()

    # Cross
    cross_ref = c << gf.get_component(greek_cross_component)
    cross_ref.x = (
        2 * pad_spacing - (pad_spacing - gf.get_component(pad).info["size"][0]) / 2
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
        pad_ref.x = index * pad_spacing + pad_ref.xsize / 2
        via_ref = c << gf.get_component(pad_via)
        if index < 2:
            via_ref.connect("e2", destination=pad_ref.ports["e4"])
        else:
            via_ref.connect("e4", destination=pad_ref.ports["e2"])

        route = gf.routing.get_route(
            cross_ref[cross_pad_via_port_pairs[index][0]],
            via_ref[cross_pad_via_port_pairs[index][1]],
            cross_section=xs_metal,
            bend=gf.c.wire_corner,
            start_straight_length=5,
            end_straight_length=5,
        )
        c.add(route.references)

    return c


@gf.cell
def greek_cross_offset_pads(
    cross_struct_length: float = 30.0,
    cross_struct_width: float = 1.0,
    cross_struct_layers: LayerSpecs = ("WG",),
    cross_implant_length: float = 30.0,
    cross_implant_width: float = 2.0,
    cross_implant_layers: LayerSpecs = ("N",),
    contact_layers: LayerSpecs = ("WG", "NPP"),
    contact_offset: float = 10,
    contact_buffer: float = 10,
    pad_width: float = 50,
) -> gf.Component:
    """Greek cross, with silicon islands on each side of the cross to place larger contacting regions.

    Args:
        cross_struct_length: length of structural part of cross e.g. silicon core.
        cross_struct_width: width of structural part of cross  e.g. silicon core.
        cross_struct_layers: layers to be considered "structural".
        cross_implant_length: length of implantation part of cross.
        cross_implant_width: width of implantation part of cross.
        cross_implant_layers: layers to be considered "implants".
        contact_layers: layers to include under and around the pad.
        contact_offset: fudge factor to move pad relative to cross.
        contact_buffer: amount of dopants around pad in contact.
        pad_width: pad size.

    .. code::

            pad_width
            <------->
            _________ cross_implant_length, cross_struct_length
            |       |<------->
        4x  |       |         ↓
            |       |======== cross_implant_width, cross_struct_width
            |_______|         ↑
                <-------------->
            contact_offset (fudge)

    References:
    - Walton, Anthony J.. “MICROELECTRONIC TEST STRUCTURES.” (1999).
    - W. Versnel, Analysis of the Greek cross, a Van der Pauw structure with finite
        contacts, Solid-State Electronics, Volume 22, Issue 11, 1979, Pages 911-914,
        ISSN 0038-1101, https://doi.org/10.1016/0038-1101(79)90061-3.
    - S. Enderling et al., "Sheet resistance measurement of non-standard cleanroom
        materials using suspended Greek cross test structures," IEEE Transactions on
        Semiconductor Manufacturing, vol. 19, no. 1, pp. 2-9, Feb. 2006,
        doi: 10.1109/TSM.2005.863248.

    """
    c = gf.Component()

    # Layout cross
    for layer in cross_struct_layers:
        c << gf.get_component(
            cross,
            length=2 * cross_struct_length + cross_struct_width,
            width=cross_struct_width,
            layer=layer,
        )
    for layer in cross_implant_layers:
        c << gf.get_component(
            cross,
            length=2 * cross_implant_length + cross_implant_width,
            width=cross_implant_width,
            layer=layer,
        )

    pad_offset = pad_width / 2 + cross_implant_length / 2

    # contact vias and pads
    for sgnx, sgny in product([1, -1], [1, -1]):
        pad_rotation = np.arctan2(sgny, sgnx) * 180 / np.pi - 45
        c2 = gf.Component(f"contact_{sgnx}_{sgny}")
        c2 << gf.get_component(via_stack, size=(pad_width, pad_width))
        c2 << gf.get_component(pad, size=(pad_width, pad_width))
        for layer in contact_layers:
            w = gf.get_component(
                rectangle,
                size=(
                    pad_width + contact_buffer,
                    pad_width + cross_implant_length / 2 + contact_buffer,
                ),
                layer=layer,
            )
            ref = c2 << w
            ref.move(
                np.array(
                    [
                        -1 * pad_offset + cross_implant_length / 2 - contact_buffer / 2,
                        -1 * pad_offset - contact_offset / 2,
                    ]
                )
            )
        contact = c << c2
        contact.rotate(pad_rotation)
        contact.move(np.array([sgnx * pad_offset, sgny * pad_offset]))

    return c.flatten()


if __name__ == "__main__":
    # c = greek_cross_offset_pads()
    c = greek_cross_with_pads()
    c.show(show_ports=True)
