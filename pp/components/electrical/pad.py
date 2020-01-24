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
    """ column array of rectangular pads
    """
    start = (start[0] - n * spacing[0] / 2, start[1])
    c = pp.Component()
    pad = pp.call_if_func(pad)
    parray = c.add_array(device=pad, start=start, spacing=spacing, num_devices=n)
    for port_name in port_list:
        for i, p in enumerate(parray):
            port_name_new = "{}{}".format(port_name, i)
            c.add_port(port=p.ports[port_name], name=port_name_new)
    return c


@autoname
def pad_array_xcentered(pad=pad, start=(0, 0), spacing=(150, 0), n=6, port_list=["N"]):
    start = (start[0] - n * spacing[0] / 2, start[1])
    c = pp.Component()
    pad = pp.call_if_func(pad)
    parray = c.add_array(device=pad, start=start, spacing=spacing, num_devices=n)
    for port_name in port_list:
        for i, p in enumerate(parray):
            port_name_new = "{}{}".format(port_name, i)
            c.add_port(port=p.ports[port_name], name=port_name_new)
    return c


if __name__ == "__main__":
    # c = pad()
    # c = pad(width=10, height=10)
    c = pad_array_xcentered(n=2)
    print(c.ports.keys())
    # print(c.settings['spacing'])
    pp.show(c)
