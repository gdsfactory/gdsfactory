"""menu inspired by SiEPIC tools.
"""

import pya

__version__ = "5.12.25"


def set_menu():
    menu = pya.Application.instance().main_window().menu()

    s0 = "gdsfactory"
    if not (menu.is_menu(s0)):
        menu.insert_menu("macros_menu", s0, f"gdsfactory {__version__}")
