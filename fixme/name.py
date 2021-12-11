import gdsfactory as gf

if __name__ == "__main__":
    c1 = gf.c.compass()

    @gf.cell
    def compass(layer=(2, 0)):
        return gf.c.compass(layer=layer)

    c2 = compass()
    assert c1.name != c2.name, f"{c1.name} should differ from {c2.name}"
