import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.mzi()
    c.show()
    c.write_gds("mzi.gds")
