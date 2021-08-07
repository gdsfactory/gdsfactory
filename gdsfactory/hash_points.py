import hashlib

from gdsfactory.types import Coordinates, Number


def format_float(x: Number) -> str:
    return "{:.3f}".format(x).rstrip("0").rstrip(".")


def _fmt_cp(cps: Coordinates) -> str:
    return "_".join([f"({format_float(p[0])},{format_float(p[1])})" for p in cps])


def hash_points(points: Coordinates) -> str:
    return hashlib.md5(_fmt_cp(points).encode()).hexdigest()
