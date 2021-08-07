if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    m1 = c << gf.components.mmi1x2()
    m2 = c << gf.components.mmi1x2()

    m2.reflect_h(port_name="E1")
    m2.movex(10)

    c.show()
