"""Greek cross test structure."""
from itertools import product

import numpy as np

import gdsfactory as gf
from gdsfactory.components.cross import cross
from gdsfactory.components.pad import pad
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.via_stack import via_stack
from gdsfactory.typings import LayerSpecs


@gf.cell
def greek_cross(
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
    """Process control monitor for dopant sheet resistivity and linewidth variation.

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
            c2 << w.move(
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

    return c


if __name__ == "__main__":
    c = greek_cross()
    c.show(show_ports=True)
