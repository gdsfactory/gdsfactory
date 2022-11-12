"""Stream GDS to Klayout.

You can install gdsfactory klayout integration:

- run `gf tool install`
- install the Klayout plugin through klayout package manager.
"""

import json
import os
import socket
from pathlib import Path
from typing import Union


def show(
    gds_filename: Union[Path, str], keep_position: bool = True, port: int = 8082
) -> None:
    """Show GDS in klayout.

    Args:
        gds_filename: to show.
        keep_position: keep position and active layers.
        port: klayout server port.

    """
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
    except OSError:
        pass


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    gdspath = c.write_gds()
    show(gdspath)
