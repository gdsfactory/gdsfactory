import pp
from pp.component import Component


def test_netlist_write() -> Component:
    c = pp.c.mzi()
    # netlist = c.get_netlist()
    # netlist.pop('connections')
    c.write_netlist("mzi.yml")
    # OmegaConf.save(netlist, "mzi.yml")
    return c


if __name__ == "__main__":
    c = test_netlist_write()
