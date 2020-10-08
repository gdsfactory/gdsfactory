from omegaconf import OmegaConf


def test_netlist_write():
    from pp.components.mzi import mzi

    c = mzi()
    netlist = c.get_netlist()
    # netlist.pop('connections')
    OmegaConf.save(netlist, "mzi.yml")


if __name__ == "__main__":
    c = test_netlist_write()
