import os
import pathlib
import shutil
import sys

if sys.platform == "win32":
    klayout_folder = "KLayout"
else:
    klayout_folder = ".klayout"


cwd = pathlib.Path(__file__).resolve().parent
home = pathlib.Path.home()
src = cwd / "klayout" / "pymacros" / "klive.lym"
dest = home / klayout_folder / "pymacros" / "klive.lym"
dest_folder = home / klayout_folder / "pymacros"


def install_klive():
    """ Builds and installs the extension """
    if dest.exists():
        print("klive already installed")
        return

    dest_folder.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(src, dest)
    print("klive installed to {}".format(dest))


if __name__ == "__main__":
    install_klive()
