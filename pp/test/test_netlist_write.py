from omegaconf import OmegaConf

import pp


def test_netlist_write():
    c = pp.c.mzi()
    netlist = c.get_netlist()
    # netlist.pop('connections')
    OmegaConf.save(netlist, "mzi.yml")


if __name__ == "__main__":
    c = test_netlist_write()
