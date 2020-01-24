import pp
from pp.components.grating_coupler.grating_coupler_tree import grating_coupler_tree


def shorted_pads(width=100, n_pads=8, pad_spacing=150, layer=pp.LAYER.WG):
    c = pp.Component(name="shorted_pads")
    pad = pp.c.rectangle_centered(x=width, y=width, layer=layer)
    for i in range(n_pads):
        pad_ref = c.add_ref(pad)
        pad_ref.movex(i * pad_spacing - n_pads / 2 * pad_spacing + pad_spacing / 2)

    short = pp.c.rectangle_centered(x=pad_spacing * (n_pads - 1), y=10, layer=layer)
    c.add_ref(short)
    return c


@pp.autoname
def grating_coupler_tree_with_pads():
    c = pp.Component()

    gratings = c << grating_coupler_tree()
    pads = c << shorted_pads()

    gratings.move(-gratings.size_info.center)
    pads.movey(gratings.ymax + 10 - pads.ymin)
    return c


if __name__ == "__main__":
    # c = shorted_pads()
    c = grating_coupler_tree_with_pads()
    pp.show(c)
