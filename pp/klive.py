""" updates GDS view in Klayout dynamically.
Requires the Klayout plugin installed in Klayout.
This happens when you run `bash install.sh` from the top of the gdsfactory package
"""

import json
import os
import socket
from pathlib import Path
from typing import Union


def show(gds_filename: Union[Path, str], keep_position: bool = True) -> None:
    """ Show GDS in klayout """
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
        print(
            "error sending GDS to klayout. Make sure have Klayout opened and that you have installed klive with `pf install`"
        )


if __name__ == "__main__":
    import pp

    c = pp.c.waveguide()
    gdspath = pp.write_gds(c)
    show(gdspath)
