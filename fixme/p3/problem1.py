"""How can we track connectivity?

For example, make sure the ports are connected.

"""

import gdsfactory as gf


@gf.cell
def problem1():
    c = gf.Component()
    n = 2

    ps = gf.c.mzi_phase_shifter_top_heater_metal()
    phase_shifters = c << gf.components.array(ps, rows=n, columns=1, spacing=(0, 300))
    pads = c << gf.c.pad_array(columns=2 * n)

    pads.center = (0, 0)
    phase_shifters.center = (0, 0)
    pads.ymin = phase_shifters.ymax + 100

    ports1 = pads.get_ports_list()
    ports2 = phase_shifters.get_ports_list(port_type="electrical", orientation=90)

    for port1, port2 in zip(ports1, ports2):
        c.add_label(position=port1.midpoint, text=port1.name)
        c.add_label(position=port2.midpoint, text=port1.name)

    return c


@gf.cell
def solution1():
    c = gf.Component()
    n = 2

    ps = gf.c.mzi_phase_shifter_top_heater_metal()
    phase_shifters = c << gf.components.array(ps, rows=n, columns=1, spacing=(0, 300))
    pads = c << gf.c.pad_array(columns=2 * n)

    pads.center = (0, 0)
    phase_shifters.center = (0, 0)
    pads.ymin = phase_shifters.ymax + 100

    ports1 = pads.get_ports_list()
    ports2 = phase_shifters.get_ports_list(port_type="electrical", orientation=90)

    for port1, port2 in zip(ports1, ports2):
        c.add_label(position=port1.midpoint, text=port1.name)
        c.add_label(position=port2.midpoint, text=port1.name)

    routes = gf.routing.get_bundle(
        ports1, ports2, cross_section=gf.cross_section.metal3
    )
    for route in routes:
        c.add(route.references)
    return c


if __name__ == "__main__":
    c = solution1()
    c.write_gds("solution1.gds")
    c = problem1()
    c.write_gds("problem1.gds")
    c.show()
