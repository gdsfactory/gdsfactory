import pp


def test_netlist_write():
    c = pp.c.mzi()
    # netlist = c.get_netlist()
    # netlist.pop('connections')
    c.write_netlist("mzi.yml")
    # OmegaConf.save(netlist, "mzi.yml")
    return c


if __name__ == "__main__":
    c = test_netlist_write()
