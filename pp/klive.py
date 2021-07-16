"""Update GDS view in Klayout dynamically.
Requires the Klayout plugin installed in Klayout.
This happens when you run `pf install`.
"""

import json
import os
import socket
from pathlib import Path
from typing import Union

from phidl.quickplotter import set_quickplot_options


def set_plot_options(
    show_ports: bool = True,
    show_subports: bool = False,
    label_aliases: bool = False,
    new_window: bool = False,
    blocking: bool = False,
    zoom_factor: float = 1.4,
):
    """Set plot options for matplotlib"""
    set_quickplot_options(
        show_ports=show_ports,
        show_subports=show_subports,
        label_aliases=label_aliases,
        new_window=new_window,
        blocking=blocking,
        zoom_factor=zoom_factor,
    )


def show(gds_filename: Union[Path, str], keep_position: bool = True) -> None:
    """Show GDS in klayout."""
    if not os.path.isfile(gds_filename):
        raise ValueError(f"{gds_filename} does not exist")
    data = {
        "gds": os.path.abspath(gds_filename),
        "keep_position": keep_position,
    }
    data_string = json.dumps(data)
    try:
        conn = socket.create_connection(("127.0.0.1", 8082), timeout=1.0)
        data_string = data_string + "\n"
        data_string = (
            data_string.encode() if hasattr(data_string, "encode") else data_string
        )
        conn.sendall(data_string)
        conn.close()
    except socket.error:
        pass


if __name__ == "__main__":
    import pp

    c = pp.components.straight()
    gdspath = c.write_gds()
    show(gdspath)
