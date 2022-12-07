"""Greek cross test structure."""
from itertools import product

import numpy as np

import gdsfactory as gf
from gdsfactory.components.cross import cross
from gdsfactory.components.pad import pad
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.via_stack import via_stack
from gdsfactory.tech import LAYER
from gdsfactory.types import Layers


@gf.cell
def greek_cross(
    cross_struct_length: float = 30.0,
    cross_struct_width: float = 1.0,
    cross_struct_layers: Layers = (LAYER.WG),
    cross_implant_length: float = 30.0,
    cross_implant_width: float = 2.0,
    cross_implant_layers: Layers = (LAYER.N),
    contact_layers: Layers = (LAYER.WG, LAYER.NPP),
    contact_offset: float = 10,
    contact_buffer: float = 10,
    pad_width: float = 50,
) -> gf.Component:
    """Process control monitor for dopant sheet resistivity and linewidth variation.

        pad_width
        <------->
        _________ cross_implant_length, cross_struct_length
        |       |<------->
    4x  |       |         ↓
        |       |======== cross_implant_width, cross_struct_width
        |_______|         ↑
            <-------------->
        contact_offset (fudge)

    Arguments:
        cross_struct_length (float): length of structural part of cross e.g. silicon core
        cross_struct_width (float): width of structural part of cross  e.g. silicon core
        cross_struct_layers (Layers tuple): layers to be considered "structural"
        cross_implant_length (float): length of implantation part of cross
        cross_implant_width (float): width of implantation part of cross
        cross_implant_layers (Layers tuple): layers to be considered "implants"
        contact_layers (Layer tuple): layers to include under and around the pad
        contact_offset (float): fudge factor to move pad relative to cross
        contact_buffer (float): amount of dopants around pad in contact
        pad_width (float): pad size
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

    # Layout contacting vias and pads
    for sgnx, sgny in product([1, -1], [1, -1]):
        pad_offset = pad_width / 2 + cross_implant_length / 2
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
