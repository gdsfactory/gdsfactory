if __name__ == "__main__":

    import gdsfactory as gf

    c = gf.Component("parent")
    ref = c << gf.components.straight()
    c.add_ports(ref.ports)
    ref.movex(5)
    assert c.ports["o1"].center[0] == 5, c.ports["o1"].center[0]
    c.show(show_ports=True)
