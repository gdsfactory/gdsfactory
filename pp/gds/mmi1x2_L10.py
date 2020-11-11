import pp

if __name__ == "__main__":
    c = pp.c.mmi1x2(length_mmi=10)
    pp.write_gds(c, "mmi1x2_L10.gds")
