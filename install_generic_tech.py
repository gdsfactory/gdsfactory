"""
symlink generic tech to klayout

"""

import os
import pathlib
import sys

if sys.platform == "win32":
    klayout_folder = "KLayout"
else:
    klayout_folder = ".klayout"


cwd = pathlib.Path(__file__).resolve().parent
home = pathlib.Path.home()
src = cwd / "klayout" / "tech"
dest_folder = home / klayout_folder / "tech"
dest = home / klayout_folder / "tech" / "generic"


def install_generic_tech():
    """ installs generic layermap """
    if dest.exists():
        print("generic tech already installed")
        return

    dest_folder.mkdir(exist_ok=True, parents=True)
    os.symlink(src, dest)
    print(f"generic layermap installed to {dest}")


if __name__ == "__main__":
    install_generic_tech()
