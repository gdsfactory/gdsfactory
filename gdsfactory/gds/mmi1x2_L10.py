import gdsfactory

if __name__ == "__main__":
    c = gdsfactory.components.mmi1x2(length_mmi=10)
    c.write_gds("mmi1x2_L10.gds")
