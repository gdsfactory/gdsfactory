import os
import sys


def disable_print() -> None:
    sys.stdout = open(os.devnull, "w")


def enable_print() -> None:
    sys.stdout = sys.__stdout__
