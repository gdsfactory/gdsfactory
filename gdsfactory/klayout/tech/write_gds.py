import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.mzi2x2(with_elec_connections=True)
    c.show()
    c.write_gds("mzi.gds")
