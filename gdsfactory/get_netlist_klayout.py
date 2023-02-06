from __future__ import annotations

from gdsfactory.typings import ComponentOrPath


def get_netlist_klayout(gdspath: ComponentOrPath) -> None:
    """Returns a boolean operation between two components Uses klayout python API.

    Args:
        gdspath: path to GDS.

    """
    import klayout.db as pya

    layout = pya.Layout()
    layout.read(str(gdspath))
    return layout.top_cell()


if __name__ == "__main__":
    import klayout.db as pya

    from gdsfactory.samples.demo.lvs import pads_with_routes

    c = pads_with_routes()
    c.show()
    gdspath = c.write_gds("a.gds")

    layout = pya.Layout()
    layout.read(str(gdspath))
    c = get_netlist_klayout(gdspath)
