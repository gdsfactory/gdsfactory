import os
import pathlib
import shutil
import sys

if sys.platform == "win32":
    klayout_folder = "KLayout"
else:
    klayout_folder = ".klayout"


home = pathlib.Path.home()
dest_folder = home / klayout_folder / "pymacros"
dest_folder.mkdir(exist_ok=True, parents=True)


def install_klive(src, dest):
    """ Builds and installs the extension """
    if dest.exists():
        print(f"removing klive already installed in {dest}")
        os.remove(dest)

    shutil.copy(src, dest)
    print(f"klive installed to {dest}")


if __name__ == "__main__":
    cwd = pathlib.Path(__file__).resolve().parent
    src = cwd / "klayout" / "pymacros" / "klive.lym"
    dest = dest_folder / "klive.lym"
    install_klive(src, dest)
