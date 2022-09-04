"""FIXME."""
import gdsfactory as gf


if __name__ == "__main__":
    c = gf.components.switch_tree(bend_s=None)
    n = c.get_netlist_recursive()
