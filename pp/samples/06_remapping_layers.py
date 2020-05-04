""" You can remap layers

"""


import pp

c = pp.Component("waveguides_sample")


wg1 = c << pp.c.waveguide(length=10, width=1)
wg2 = c << pp.c.waveguide(length=10, width=2, layer=pp.LAYER.SLAB90)
wg3 = c << pp.c.waveguide(length=10, width=3, layer=pp.LAYER.SLAB150)

wg2.connect(port="W0", destination=wg1.ports["E0"])
wg3.connect(port="W0", destination=wg2.ports["E0"], overlap=1)


c.remap_layers({pp.LAYER.WG: pp.LAYER.SLAB150})

pp.qp(c)  # quickplot it!
pp.show(c)  # show it in klayout
