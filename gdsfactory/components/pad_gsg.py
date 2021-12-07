"""High speed GSG pads."""

import gdsfactory as gf
from gdsfactory.components.pad import pad as pad_function
from gdsfactory.components.rectangle import rectangle
from gdsfactory.types import ComponentFactory, Float2, Layer

rectangle_m3 = gf.partial(rectangle, layer=gf.LAYER.M3)


@gf.cell
def pad_gsg_short(
    contact: ComponentFactory = rectangle_m3,
    size: Float2 = (22, 7),
    layer_metal: Layer = gf.LAYER.M3,
    metal_spacing: float = 5.0,
    short: bool = True,
    pad: ComponentFactory = pad_function,
    pad_spacing: float = 150,
):
    """Returns high speed GSG pads for calibrating the RF probes.

    Args:
        contact: where the RF pads connect to
        size: for the contact
        layer_metal:
        metal_spacing:
        short: if False returns an open
        pad: function for pad
        pad_spacing: in um
    """
    c = gf.Component()
    sig = gf.c.rectangle(size=size, layer=layer_metal).ref()

    gnd_top = c << contact(size=size)
    sig = contact(size=size)
    if short:
        sig = c << sig
    gnd_bot = c << contact(size=size)

    gnd_bot.ymax = sig.ymin
    gnd_top.ymin = sig.ymax

    gnd_top.movex(-metal_spacing)
    gnd_bot.movex(-metal_spacing)

    pads = c << gf.c.array(pad(), columns=1, rows=3, spacing=(0, pad_spacing))
    pads.xmin = sig.xmax + 50
    pads.y = 0

    c << gf.routing.route_quad(
        gnd_bot.ports["e4"], pads.ports["e1_1_1"], layer=layer_metal
    )
    c << gf.routing.route_quad(
        gnd_top.ports["e2"], pads.ports["e1_3_1"], layer=layer_metal
    )
    c << gf.routing.route_quad(sig.ports["e3"], pads.ports["e1_2_1"], layer=layer_metal)
    return c


pad_gsg_open = gf.partial(pad_gsg_short, short=False)


if __name__ == "__main__":
    # c = pad_array_double()
    # c = pad_gsg_short()
    c = pad_gsg_open()
    c.show()
