import hashlib
from typing import Tuple

Floats = Tuple[float, ...]


def format_float(x: float) -> str:
    return "{:.3f}".format(x).rstrip("0").rstrip(".")


def _fmt_cp(cps: Floats) -> str:
    return "_".join([f"({format_float(p[0])},{format_float(p[1])})" for p in cps])


def hash_points(points: Floats) -> str:
    return hashlib.md5(_fmt_cp(points).encode()).hexdigest()
