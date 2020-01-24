import os
import pathlib
import shutil
import sys

if sys.platform == "win32":
    KLAYOUT_FOLDER = "KLayout"
else:
    KLAYOUT_FOLDER = ".klayout"


CWD = pathlib.Path(__file__).resolve().parent
HOME = pathlib.Path.home()
MACRO_PATH = CWD / "klive.lym"
MACROS_PATH = HOME / KLAYOUT_FOLDER / "pymacros"


def install_klive():
    """ Builds and installs the extension """
    if os.path.exists(MACRO_PATH):
        print("klive already installed")
        return

    # Make build directory
    if not os.path.exists(MACROS_PATH):
        os.makedirs(MACROS_PATH)

    # Try to copy to the right place
    shutil.copyfile(MACRO_PATH, MACROS_PATH)
    print("klive installed to {}".format(MACRO_PATH))


if __name__ == "__main__":
    install_klive()
