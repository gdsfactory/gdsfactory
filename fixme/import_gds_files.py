if __name__ == "__main__":
    import gdsfactory as gf

    c1 = gf.components.mzi()
    c2 = gf.components.mzi(delta_length=100, name="mzi")
    gdspath1 = c1.write_gds("a.gds")
    gdspath2 = c2.write_gds("b.gds")

    c = gf.Component("compare")
    ref1 = c << gf.import_gds(gdspath1)
    ref2 = c << gf.import_gds(gdspath2)
    ref2.ymin = ref1.ymax
    c.show()
