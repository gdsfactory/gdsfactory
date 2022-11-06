import gdsfactory as gf


@gf.cell
def test():
    c = gf.Component()
    bend = c << gf.components.bend_circular()
    bend.rotate(-90)
    c.add_ports(bend.ports)
    c.absorb(bend)
    return c


if __name__ == "__main__":
    c = gf.Component("demo")
    bend = c << test()
    c.add_ports(bend.ports)
    c.show(show_ports=True)
