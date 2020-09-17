"""
updates GDS view in Klayout dynamically.
Requires the Klayout plugin installed in Klayout.
This happens when you run `bash install.sh` from the top of the gdsfactory package
"""

import os
import socket
import json
from pathlib import PosixPath


def show(gds_filename: PosixPath, keep_position: bool = True) -> None:
    """ Show GDS in klayout """
    if not os.path.isfile(gds_filename):
        raise ValueError("{} does not exist".format(gds_filename))
    data = {
        "gds": os.path.abspath(gds_filename),
        "keep_position": keep_position,
    }
    data = json.dumps(data)
    try:
        conn = socket.create_connection(("127.0.0.1", 8082), timeout=0.5)
        data = data + "\n"
        data = data.encode() if hasattr(data, "encode") else data
        conn.sendall(data)
        conn.close()
    except socket.error:
        print("warning, could not connect to the klive server")
        pass
