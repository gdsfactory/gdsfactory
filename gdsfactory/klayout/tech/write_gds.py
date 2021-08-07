import gdsfactory

if __name__ == "__main__":
    c = gdsfactory.components.mzi2x2(with_elec_connections=True)
    c.show()
    c.write_gds("mzi.gds")
