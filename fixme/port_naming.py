"""Port naming behaves weird in some cases

"""

import gdsfactory as gf


if __name__ == "__main__":

    c = gf.Component()
    w = gf.components.straight_heater_doped_strip(length=80)

    w1 = c << w
    w2 = c << w
    w2.movey(120)
    w2.movex(20)

    c.add_ports(w1.ports, prefix="bot")
    c.add_ports(w2.ports, prefix="top")

    c.auto_rename_ports()
    c.show()
