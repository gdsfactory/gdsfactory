"""Mim capacitor."""

import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.components.via_stack import via_stack_m2_m3
from gdsfactory.typings import Component, LayerSpecs


@gf.cell
def mimcap(
    size: tuple[float, float] = (64, 64),
    width_contact: float = 2.5,
    gap_top_metal: float = 2,
    via_stack=via_stack_m2_m3,
    layers_bot: LayerSpecs = ("M2",),
    offsets_bot: tuple[float, ...] = (0.5,),
) -> Component:
    """Returns high speed GSG pads for calibrating the RF probes.

    By default contact is on the left

    Args:
        size: for the mimcap. (right).
        width_contact: width of the contact (left).
        gap_top_metal: between left and right sides.
    """
    c = Component()

    size_contact = (width_contact, size[1])
    size_mimcap = (size[0] - width_contact - gap_top_metal - 2.0, size[1] - 2.0)

    contact = c << via_stack(size=size_contact)
    mim = c << via_stack(size=size_mimcap)
    mim.xmin = contact.xmax + gap_top_metal

    c2 = c.copy()

    for layer, offset in zip(layers_bot, offsets_bot):
        points = get_padding_points(c2, default=offset)
        c.add_polygon(points, layer=layer)
    c.add_ports(contact.ports, prefix="l_")
    c.add_ports(mim.ports, prefix="r_")
    return c


if __name__ == "__main__":
    c = mimcap(size=(20, 10), width_contact=2)
    gf.remove_from_cache(c)
    c.show(show_ports=True)
