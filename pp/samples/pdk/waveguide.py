import pp
from layers import LAYER


@pp.autoname
def waveguide(layer=LAYER.WG, layers_cladding=[], **kwargs):
    c = pp.c.waveguide(layer=layer, layers_cladding=layers_cladding, **kwargs)
    return c


if __name__ == "__main__":
    c = waveguide()
    pp.show(c)
