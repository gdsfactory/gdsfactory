""" symlink generic tech to klayout
"""

import os
import pathlib
import sys


def symlink(src, dest):
    """ installs generic layermap """
    if dest.exists():
        print("generic tech already installed")
        return

    dest_folder = dest.parent
    dest_folder.mkdir(exist_ok=True, parents=True)
    try:
        os.symlink(src, dest)
    except Exception:
        os.remove(dest)
        os.symlink(src, dest)
    print(f"added symlink from {src} to {dest}")


if __name__ == "__main__":
    if sys.platform == "win32":
        klayout_folder = "KLayout"
    else:
        klayout_folder = ".klayout"

    cwd = pathlib.Path(__file__).resolve().parent
    home = pathlib.Path.home()
    src = cwd / "klayout" / "tech"
    dest = home / klayout_folder / "tech" / "generic"

    symlink(src, dest)

    src = cwd / "klayout" / "drc" / "generic.lydrc"
    dest = home / klayout_folder / "drc" / "generic.lydrc"
    symlink(src, dest)
