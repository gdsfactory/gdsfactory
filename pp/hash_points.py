import hashlib
from typing import List, Tuple, Union

from numpy import float64, ndarray


def format_float(x: Union[float, int, float64]) -> str:
    return "{:.3f}".format(x).rstrip("0").rstrip(".")


def _fmt_cp(
    cps: Union[
        List[ndarray],
        List[Union[Tuple[int, int], Tuple[float, int], Tuple[float, float]]],
        List[Tuple[float, float]],
        List[Union[Tuple[int, int], Tuple[float, int], Tuple[float, float64]]],
        List[Union[ndarray, Tuple[float64, float64], Tuple[float, float]]],
    ]
) -> str:
    return "_".join([f"({format_float(p[0])},{format_float(p[1])})" for p in cps])


def hash_points(
    points: Union[
        List[ndarray],
        List[Union[Tuple[int, int], Tuple[float, int], Tuple[float, float]]],
        List[Tuple[float, float]],
        List[Union[Tuple[int, int], Tuple[float, int], Tuple[float, float64]]],
        List[Union[ndarray, Tuple[float64, float64], Tuple[float, float]]],
    ]
) -> str:
    return hashlib.md5(_fmt_cp(points).encode()).hexdigest()
