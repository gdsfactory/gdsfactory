import hashlib


def format_float(x):
    return "{:.3f}".format(x).rstrip("0").rstrip(".")


def _fmt_cp(cps):
    return "_".join([f"({format_float(p[0])},{format_float(p[1])})" for p in cps])


def hash_points(points):
    return hashlib.md5(_fmt_cp(points).encode()).hexdigest()
