import gdsfactory as gf

from gpdk import components


@gf.cell
def mzi_heater_python(dL: float = 10, length_heater: float = 100) -> gf.Component:
    """MZI phase shifter with heater.

    Args:
        dL: delta length between the two arms.
        length_heater: length of the heater.
    """
    delta_length = dL
    c = gf.Component()
    sp = c << gf.c.mmi2x2()
    cp = c << gf.c.mmi2x2()

    b1 = c << gf.c.bend_euler()
    b1.connect("o1", sp.ports["o3"])

    sl = c << gf.c.straight(length=delta_length / 2)
    sl.name = "sl"
    sl.connect("o1", b1.ports["o2"])

    b2 = c << gf.c.bend_euler()
    b2.connect("o2", sl.ports["o2"])

    h = c << components.straight_heater_metal(length=length_heater)
    h.connect("o1", b2.ports["o1"])

    b3 = c << gf.c.bend_euler()
    b3.connect("o2", h.ports["o2"])

    sr = c << gf.c.straight(length=delta_length / 2)
    sr.name = "sr"
    sr.connect("o1", b3.ports["o1"])

    b4 = c << gf.c.bend_euler()
    b4.connect("o1", sr.ports["o2"])
    sp.connect("o2", b4.ports["o2"])

    gf.routing.route_bundle(
        c,
        [sp.ports["o1"]],
        [cp.ports["o4"]],
        cross_section="strip",
    )

    c.add_ports(sp.ports.filter(orientation=0))
    c.add_ports(cp.ports.filter(orientation=180))

    return c
