import gdsfactory as gf
from gdsfactory.component import Component


def test_netlist_write() -> Component:
    c = gf.components.mzi()
    # netlist = c.get_netlist()
    # netlist.pop('connections')
    c.write_netlist("mzi.yml")
    # OmegaConf.save(netlist, "mzi.yml")
    return c


if __name__ == "__main__":
    c = test_netlist_write()
    c.show(show_ports=True)
