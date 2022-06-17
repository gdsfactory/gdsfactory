"""
Sample of a reticle
"""

import gdsfactory as gf


def mzi_te_pads1(**kwargs):
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_single(c)
    c = c.rotate(-90)
    c = gf.routing.add_electrical_pads_top(c)
    gf.add_labels.add_labels_to_ports_electrical(component=c, prefix=f"elec-{c.name}-")
    return c


def mzi_te_pads2(**kwargs):
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_single(c)
    c = c.rotate(-90)
    c = gf.routing.add_electrical_pads_top_dc(c)
    gf.add_labels.add_labels_to_ports_electrical(component=c, prefix=f"elec-{c.name}-")
    return c


def mzi_te_pads3(**kwargs):
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_single(c)
    c = c.rotate(-90)
    c = gf.routing.add_electrical_pads_shortest(c)
    gf.add_labels.add_labels_to_ports_vertical_dc(component=c, prefix=f"elec-{c.name}-")
    return c


if __name__ == "__main__":
    c = mzi_te_pads3()
    c.show()

    # c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    # c = gf.routing.add_fiber_single(c)
    # c = c.rotate(-90)
    # c = gf.routing.add_electrical_pads_top(c)
    # gf.add_labels.add_labels_to_ports_electrical(component=c, prefix=f"elec-{c.name}")
    # c.show()
    print(c.get_labels())
    # print(c.get_ports_dict(port_type="electrical").keys())
