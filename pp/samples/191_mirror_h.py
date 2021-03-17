if __name__ == "__main__":
    import pp

    c = pp.Component()
    m1 = c << pp.components.mmi1x2()
    m2 = c << pp.components.mmi1x2()

    m2.reflect_h(port_name="E1")
    m2.movex(10)

    c.show()
