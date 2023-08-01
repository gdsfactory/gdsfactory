"""Disable print statements for overly verbose simulators."""
from __future__ import annotations

import os
import sys


def disable_print() -> None:
    sys.stdout = open(os.devnull, "w")


def enable_print() -> None:
    sys.stdout = sys.__stdout__
