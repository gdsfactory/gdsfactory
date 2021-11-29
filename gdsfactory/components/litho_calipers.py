from typing import Tuple

import gdsfactory as gf
from gdsfactory import components as pc
from gdsfactory.component import Component


@gf.cell
def litho_calipers(
    notch_size: Tuple[float, float] = (2.0, 5.0),
    notch_spacing: float = 2.0,
    num_notches: int = 11,
    offset_per_notch: float = 0.1,
    row_spacing: float = 0.0,
    layer1: Tuple[int, int] = (1, 0),
    layer2: Tuple[int, int] = (2, 0),
) -> Component:
    """Vernier caliper structure to test lithography alignment
    Only the middle finger is aligned and the rest are offset.

    adapted from phidl

    Args:
        notch_size: [xwidth, yheight]
        notch_spacing: 2
        num_notches: 11
        offset_per_notch: 0.1
        row_spacing: 0
        layer1: 1
        layer2: 2

    """

    D = gf.Component()
    num_notches_total = num_notches * 2 + 1
    centre_notch = num_notches
    R1 = pc.rectangle(size=notch_size, layer=layer1)
    R2 = pc.rectangle(size=notch_size, layer=layer2)
    for i in range(num_notches_total):
        if i == centre_notch:
            D.add_ref(R1).movex(i * (notch_size[0] + notch_spacing)).movey(
                notch_size[1]
            )
            D.add_ref(R2).movex(
                i * (notch_size[0] + notch_spacing)
                + offset_per_notch * (centre_notch - i)
            ).movey(-2 * notch_size[1] - row_spacing)
        D.add_ref(R1).movex(i * (notch_size[0] + notch_spacing))
        D.add_ref(R2).movex(
            i * (notch_size[0] + notch_spacing) + offset_per_notch * (centre_notch - i)
        ).movey(-notch_size[1] - row_spacing)

    return D


if __name__ == "__main__":
    c = litho_calipers()
    c.show()
