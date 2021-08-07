if __name__ == "__main__":
    import gdsfactory

    c = gdsfactory.Component()
    m1 = c << gdsfactory.components.mmi1x2()
    m2 = c << gdsfactory.components.mmi1x2()

    m2.reflect_h(port_name="E1")
    m2.movex(10)

    c.show()
