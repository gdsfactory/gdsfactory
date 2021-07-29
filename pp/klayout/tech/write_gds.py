import pp

if __name__ == "__main__":
    c = pp.c.mzi2x2(with_elec_connections=True)
    c.show()
    c.write_gds("mzi.gds")
