"""Based on Gerber file spec.

https://www.ucamco.com/files/downloads/file_en/456/gerber-layer-format-specification-revision-2022-02_en.pdf.

See Also:
- https://github.com/opiopan/pcb-tools-extension
- https://github.com/jamesbowman/cuflow/blob/master/gerber.py
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from gdsfactory import Component
from gdsfactory.typings import Size

PolygonPoints = Sequence[Sequence[float]] | npt.NDArray[np.floating]


class GerberLayer(BaseModel):
    name: str
    function: list[str]
    polarity: Literal["Positive", "Negative"]


class GerberOptions(BaseModel):
    header: list[str] | None = None
    mode: Literal["mm", "in"] = "mm"
    resolution: float = 1e-6
    int_size: int = 4


# For generating a gerber job json file
class BoardOptions(BaseModel):
    size: Size | None = None
    n_layers: int = 2


resolutions = {1e-3: 3, 1e-4: 4, 1e-5: 5, 1e-6: 6}


def decimal_digits(resolution: float) -> int:
    """Return Gerber decimal digits for a linear resolution.

    Args:
        resolution: smallest distance represented in the file, in file units.

    Returns:
        Number of decimal digits in the `%FS` command.

    Raises:
        ValueError: If `resolution` is not one of the supported values.
    """
    if resolution not in resolutions:
        supported = ", ".join(str(value) for value in sorted(resolutions, reverse=True))
        raise ValueError(
            f"Unsupported Gerber resolution {resolution}. Supported values: {supported}."
        )
    return resolutions[resolution]


def format_specification(int_size: int, digits: int) -> str:
    """Return a spec-compliant Gerber `%FS` command.

    Ucamco FS is `%FSLAX<int><dec>Y<int><dec>*%` (leading zeros omitted, absolute).
    """
    return f"%FSLAX{int_size}{digits}Y{int_size}{digits}*%\n"


def number(n: float, decimal_digits: int = 4) -> str:
    """Format a coordinate for Gerber using the `%FS` decimal count.

    Leading zeros are omitted (`L` in `%FSLA`). The scale is `10 ** decimal_digits`
    so coordinates match the format specifier instead of a hardcoded 10_000.

    Args:
        n: Coordinate in the same unit as `%MO` (typically Component user units).
        decimal_digits: Decimal digits from the `%FS` command.

    Returns:
        Digit string without a unit prefix, including a leading `-` when negative.
    """
    scaled_value = round(n * 10**decimal_digits)
    sign = "-" if scaled_value < 0 else ""
    return f"{sign}{abs(scaled_value)}"


def _as_xy(pp: PolygonPoints) -> list[tuple[float, float]]:
    """Normalize polygon vertices to a list of float pairs.

    `Component.get_polygons_points()` returns numpy arrays. Gerber helpers used
    to iterate them as if they were Python lists of tuples, which raises.
    """
    arr = np.asarray(pp, dtype=float)
    if arr.ndim != 2 or arr.shape[-1] != 2:
        raise ValueError(f"Expected Nx2 point array, got shape {arr.shape}")
    return [(float(x), float(y)) for x, y in arr]


def points(pp: PolygonPoints, decimal_digits: int = 4) -> str:
    xy = _as_xy(pp)
    if not xy:
        return ""
    first_x, first_y = xy[0]
    parts = [
        f"X{number(first_x, decimal_digits)}Y{number(first_y, decimal_digits)}D02*\n"
    ]
    parts.extend(
        f"X{number(x, decimal_digits)}Y{number(y, decimal_digits)}D01*\n"
        for x, y in xy[1:]
    )
    return "".join(parts)


def rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    decimal_digits: int = 4,
) -> str:
    return "D10*\n" + points(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
        decimal_digits=decimal_digits,
    )


def linestring(pp: PolygonPoints, decimal_digits: int = 4) -> str:
    return "D10*\n" + points(pp, decimal_digits=decimal_digits)


def polygon(pp: PolygonPoints, decimal_digits: int = 4) -> str:
    return "G36*\n" + points(pp, decimal_digits=decimal_digits) + "G37*\n\n"


def to_gerber(
    component: Component,
    dirpath: Path,
    layermap_to_gerber_layer: dict[tuple[int, int], GerberLayer],
    options: GerberOptions | None = None,
) -> None:
    """Writes each layer to a different Gerber file.

    Args:
        component: to export.
        dirpath: directory path.
        layermap_to_gerber_layer: map of GDS layer to GerberLayer.
        options: to save.
            header: List[str] | None = None
            mode: Literal["mm", "in"] = "mm"
            resolution: float = 1e-6
            int_size: int = 4
    """
    options = options or GerberOptions()
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    digits = decimal_digits(options.resolution)

    # Keys must match layermap tuples. Default `by="index"` uses klayout layer
    # indexes, so exported layers silently wrote empty files (#4748).
    layer_to_polygons = component.get_polygons_points(by="tuple")

    for layer_tup, layer in layermap_to_gerber_layer.items():
        filename = (dirpath / layer.name.replace(" ", "_")).with_suffix(".gbr")

        with open(filename, "w") as f:
            header = options.header or [
                "Gerber file generated by gdsfactory",
                f"Component: {component.name}",
            ]

            f.write("%TF.FileFunction," + ",".join(layer.function) + "*%\n")
            f.write(f"%TF.FilePolarity,{layer.polarity}*%\n")
            f.write(format_specification(options.int_size, digits))
            f.writelines([f"G04 {line}*\n" for line in header])

            units = options.mode.upper()
            f.write(f"%MO{units}*%\n")
            f.write("%LPD*%\n")
            f.write("G01*\n")
            f.write("%ADD10C,0.050000*%\n")

            if layer_tup in layer_to_polygons:
                f.writelines(
                    polygon(poly, decimal_digits=digits)
                    for poly in layer_to_polygons[layer_tup]
                )

            f.write("M02*\n")
