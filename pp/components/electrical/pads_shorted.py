from pp.component import Component
from pp.name import autoname
from pp.layers import LAYER
from pp.components.rectangle import rectangle_centered


@autoname
def pads_shorted(width=100, n_pads=8, pad_spacing=150, layer=LAYER.M1):
    c = Component(name="shorted_pads")
    pad = rectangle_centered(x=width, y=width, layer=layer)
    for i in range(n_pads):
        pad_ref = c.add_ref(pad)
        pad_ref.movex(i * pad_spacing - n_pads / 2 * pad_spacing + pad_spacing / 2)

    short = rectangle_centered(x=pad_spacing * (n_pads - 1), y=10, layer=layer)
    c.add_ref(short)
    return c


if __name__ == "__main__":
    import pp

    c = pads_shorted()
    pp.show(c)
