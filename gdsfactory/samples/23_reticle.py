"""Sample of a reticle top level Component."""

import gdsfactory as gf
from gdsfactory.types import Component


def mzi_te_pads1(**kwargs) -> Component:
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_single(c)
    c = c.rotate(-90)
    c = gf.routing.add_electrical_pads_top(c)
    gf.add_labels.add_labels_to_ports_electrical(component=c, prefix=f"elec-{c.name}-")
    return c


def mzi_te_pads2(**kwargs) -> Component:
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_single(c)
    c = c.rotate(-90)
    c = gf.routing.add_electrical_pads_top_dc(c)
    gf.add_labels.add_labels_to_ports_electrical(component=c, prefix=f"elec-{c.name}-")
    return c


def mzi_te_pads3(**kwargs) -> Component:
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_single(c)
    c = c.rotate(-90)
    c = gf.routing.add_electrical_pads_shortest(c)
    gf.add_labels.add_labels_to_ports_vertical_dc(component=c, prefix=f"elec-{c.name}-")
    return c


if __name__ == "__main__":
    # c = mzi_te_pads3()
    # c.show(show_ports=True)

    gc = gf.c.grating_coupler_elliptical_tm()
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_single(
        c, get_input_label_text_function=None, grating_coupler=gc
    )
    c = c.rotate(-90)
    c = gf.routing.add_electrical_pads_top(c)
    gf.add_labels.add_labels_to_ports_electrical(component=c, prefix=f"elec-{c.name}-")
    gf.add_labels.add_labels_to_ports(
        component=c, port_type="loopback", prefix=f"opttm1500-{c.name}-"
    )
    gf.add_labels.add_labels_to_ports(
        component=c, port_type="vertical_tm", prefix=f"opttm1500-{c.name}-"
    )
    c.show(show_ports=True)
