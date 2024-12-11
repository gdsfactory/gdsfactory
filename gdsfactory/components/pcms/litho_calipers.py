from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec, Size


@gf.cell
def litho_calipers(
    notch_size: Size = (2.0, 5.0),
    notch_spacing: float = 2.0,
    num_notches: int = 11,
    offset_per_notch: float = 0.1,
    row_spacing: float = 0.0,
    layer1: LayerSpec = "WG",
    layer2: LayerSpec = "SLAB150",
) -> Component:
    """Vernier caliper structure to test lithography alignment.

    Only the middle finger is aligned and the rest are offset.
    adapted from phidl

    Args:
        notch_size: [xwidth, yheight].
        notch_spacing: in um.
        num_notches: number of notches.
        offset_per_notch: in um.
        row_spacing: 0
        layer1: layer.
        layer2: layer.
    """
    D = gf.Component()
    num_notches_total = num_notches * 2 + 1
    centre_notch = num_notches
    R1 = gf.c.rectangle(size=notch_size, layer=layer1)
    R2 = gf.c.rectangle(size=notch_size, layer=layer2)

    for i in range(num_notches_total):
        if i == centre_notch:
            ref = D.add_ref(R1)
            ref.dmovex(i * (notch_size[0] + notch_spacing)).dmovey(notch_size[1])
            ref = D.add_ref(R2)
            ref.dmovex(
                i * (notch_size[0] + notch_spacing)
                + offset_per_notch * (centre_notch - i)
            ).dmovey(-2 * notch_size[1] - row_spacing)
        ref = D.add_ref(R1)
        ref.dmovex(i * (notch_size[0] + notch_spacing))
        ref = D.add_ref(R2)
        ref.dmovex(
            i * (notch_size[0] + notch_spacing) + offset_per_notch * (centre_notch - i)
        )
        ref.dmovey(-notch_size[1] - row_spacing)
    return D


if __name__ == "__main__":
    c = litho_calipers()
    c.show()
