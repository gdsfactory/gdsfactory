from pp.cell import cell
from pp.component import Component
from pp.components.rectangle import rectangle
from pp.layers import LAYER


@cell
def pads_shorted(width=100, n_pads=8, pad_spacing=150, layer=LAYER.M1):
    c = Component(name="shorted_pads")
    pad = rectangle(size=(width, width), layer=layer, centered=True)
    for i in range(n_pads):
        pad_ref = c.add_ref(pad)
        pad_ref.movex(i * pad_spacing - n_pads / 2 * pad_spacing + pad_spacing / 2)

    short = rectangle(size=(pad_spacing * (n_pads - 1), 10), layer=layer, centered=True)
    c.add_ref(short)
    return c


if __name__ == "__main__":
    import pp

    c = pads_shorted()
    pp.show(c)
