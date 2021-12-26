"""Update GDS view in Klayout dynamically.
Requires the Klayout plugin installed in Klayout.
This happens when you run `pf install`.
"""

import json
import os
import socket
from pathlib import Path
from typing import Union


def show(
    gds_filename: Union[Path, str], keep_position: bool = True, port: int = 8082
) -> None:
    """Show GDS in klayout."""
    if not os.path.isfile(gds_filename):
        raise ValueError(f"{gds_filename} does not exist")
    data = {
        "gds": os.path.abspath(gds_filename),
        "keep_position": keep_position,
    }
    data_string = json.dumps(data)
    try:
        conn = socket.create_connection(("127.0.0.1", port), timeout=1.0)
        data_string = data_string + "\n"
        data_string = (
            data_string.encode() if hasattr(data_string, "encode") else data_string
        )
        conn.sendall(data_string)
        conn.close()
    except socket.error:
        pass


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    gdspath = c.write_gds()
    show(gdspath)
