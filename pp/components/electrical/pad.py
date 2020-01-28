from pp.name import autoname
from pp.layers import LAYER
from pp.components.compass import compass
import pp

WIRE_WIDTH = 10.0


@autoname
def pad(width=100.0, height=100.0, layer=LAYER.M3):
    """ rectangular pad
    """
    c = pp.Component()
    _c = compass(size=(width, height), layer=layer).ref()
    c.add(_c)
    c.absorb(_c)
    c.ports = _c.ports
    return c


@autoname
def pad_array(pad=pad, start=(0, 0), spacing=(150, 0), n=6, port_list=["N"]):
    """ array of rectangular pads
    """
    c = pp.Component()

    for i in range(n):
        p = c << pp.call_if_func(pad)
        p.x = i * spacing[0]
        for port_name in port_list:
            port_name_new = "{}{}".format(port_name, i)
            c.add_port(port=p.ports[port_name], name=port_name_new)
    return c


if __name__ == "__main__":
    # c = pad()
    # c = pad(width=10, height=10)
    # print(c.ports.keys())
    # print(c.settings['spacing'])
    c = pad_array()
    pp.show(c)
