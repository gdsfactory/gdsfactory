if __name__ == "__main__":
    import phidl.device_layout as pd
    import phidl.geometry as pg

    nm = 1e-3

    c = pd.Device()
    r1 = c << pg.compass(size=(1, 1))
    r1.movey(0.5 * nm)

    r2 = c << pg.compass(size=(1, 1))
    r2.connect("E", r1.ports["W"])

    c.write_gds("a.gds")
    import gdsfactory as gf

    gf.show("a.gds")
