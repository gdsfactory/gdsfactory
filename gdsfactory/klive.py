"""Stream GDS to Klayout. Requires gdsfactory KLayout integration."""

from __future__ import annotations

import json
import os
import pathlib
import socket
from pathlib import Path
from typing import Optional, Union
from gdsfactory import config


def show(
    gds_filename: Union[Path, str],
    keep_position: bool = True,
    technology: Optional[str] = None,
    port: int = 8082,
    delete: bool = False,
) -> None:
    """Show GDS in klayout.

    Args:
        gds_filename: to show.
        keep_position: keep position and active layers.
        technology: Name of the KLayout technology to use.
        port: klayout server port.
        delete: deletes file.
    """
    if not os.path.isfile(gds_filename):
        raise ValueError(f"{gds_filename} does not exist")

    gds_filename = pathlib.Path(gds_filename)

    data = {
        "gds": os.path.abspath(gds_filename),
        "keep_position": keep_position,
        "technology": technology,
    }
    data_string = json.dumps(data)
    try:
        conn = socket.create_connection(("127.0.0.1", port), timeout=0.5)
        data = data_string + "\n"
        enc_data = data.encode()
        conn.sendall(enc_data)
        conn.settimeout(5)
    except OSError:
        config.logger.warning(
            "Could not connect to klive server. Is klayout open and klive plugin installed?"
        )
    else:
        msg = ""
        try:
            msg = conn.recv(1024).decode("utf-8")
            config.logger.info(f"Message from klive: {msg}")
        except OSError:
            config.logger.warning("klive didn't send data, closing")
        finally:
            conn.close()

    if delete:
        Path(gds_filename).unlink()


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.components.mzi()
    c = gf.components.straight(length=10)
    # gdspath = c.write_gds()
    # show(gdspath)
    c.show()
