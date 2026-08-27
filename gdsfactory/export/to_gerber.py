"""Gerber X2 export for handing a GDSFactory layout to RF/PCB CAM.

GDSFactory coordinates are um. Ucamco Gerber `%MO` is mm or inches. This
exporter converts layout units into file units, writes one `.gbr` per layer,
and emits the `.gbrjob` JSON the unused `BoardOptions` stub was meant for
(Ucamco Gerber Job File, revision 2020.08; partial jobs are allowed).

Spec: https://www.ucamco.com/files/downloads/file_en/456/gerber-layer-format-specification-revision-2022-02_en.pdf
Job:  https://www.ucamco.com/en/gerber/gerber-job-file

See Also:
- https://github.com/opiopan/pcb-tools-extension
- https://github.com/jamesbowman/cuflow/blob/master/gerber.py
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from gdsfactory import Component
from gdsfactory.config import __version__
from gdsfactory.typings import LayerSpec, Size

PolygonPoints = Sequence[Sequence[float]] | npt.NDArray[np.floating]
LayoutUnit = Literal["um", "mm", "in"]

# Millimetres per layout/file unit. Job-file Size is always mm (Ucamco).
_MM = {"um": 1e-3, "mm": 1.0, "in": 25.4}


class GerberLayer(BaseModel):
    name: str
    function: list[str]
    polarity: Literal["Positive", "Negative"]


class GerberOptions(BaseModel):
    header: list[str] | None = None
    mode: Literal["mm", "in"] = "mm"
    layout_unit: LayoutUnit = "um"
    resolution: float = 1e-6
    int_size: int = 4


class BoardOptions(BaseModel):
    """Ucamco job-file extras. Size is millimetres; omit to use the bbox."""

    size: Size | None = None
    n_layers: int | None = None


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


def file_unit_scale(layout_unit: LayoutUnit, mode: Literal["mm", "in"]) -> float:
    """Scale from Component user units into Gerber `%MO` units."""
    return _MM[layout_unit] / _MM[mode]


def number(n: float, decimal_digits: int = 4) -> str:
    """Format a coordinate for Gerber using the `%FS` decimal count.

    Leading zeros are omitted (`L` in `%FSLA`). The scale is `10 ** decimal_digits`
    so coordinates match the format specifier instead of a hardcoded 10_000.

    Args:
        n: Coordinate in `%MO` file units.
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
    xy = _as_xy(pp)
    if xy and xy[0] != xy[-1]:
        xy.append(xy[0])
    return "G36*\n" + points(xy, decimal_digits=decimal_digits) + "G37*\n\n"


def _gerber_filename(name: str) -> str:
    """Return the `.gbr` filename for a Gerber layer name.

    `Path.with_suffix` reads a dotted stackup name like `L1.2 signal` as having
    suffix `.2 signal` and rewrites it to `L1.gbr`, so `L1.2` and `L1.3` would
    both land on `L1.gbr` and one layer would overwrite the other.
    """
    return f"{name.replace(' ', '_')}.gbr"


def _job_file(
    component: Component,
    gerber_files: list[tuple[str, GerberLayer]],
    options: GerberOptions,
    board: BoardOptions | None,
) -> dict[str, object]:
    """Minimal CAD job file: size, layer count, file attributes (Ucamco §3.1)."""
    board = board or BoardOptions()
    to_mm = _MM[options.layout_unit]
    size = board.size or (component.xsize * to_mm, component.ysize * to_mm)
    copper_layers = [
        layer
        for _, layer in gerber_files
        if layer.function and layer.function[0].lower() == "copper"
    ]
    layer_number = board.n_layers or max(len(copper_layers), 1)
    return {
        "Header": {
            "GenerationSoftware": {
                "Vendor": "gdsfactory",
                "Application": "gdsfactory",
                "Version": __version__,
            },
            "CreationDate": datetime.now(UTC).replace(microsecond=0).isoformat(),
        },
        "GeneralSpecs": {
            "Part": "Single",
            "ProjectId": {"Name": component.name or "gdsfactory"},
            "Size": {"X": float(size[0]), "Y": float(size[1])},
            "LayerNumber": layer_number,
        },
        "FilesAttributes": [
            {
                "Path": filename,
                "FileFunction": ",".join(layer.function),
                "FilePolarity": layer.polarity,
                "FileFormat": "Gerber",
            }
            for filename, layer in gerber_files
        ],
    }


def to_gerber(
    component: Component,
    dirpath: Path,
    layermap_to_gerber_layer: dict[LayerSpec, GerberLayer],
    options: GerberOptions | None = None,
    board: BoardOptions | None = None,
    write_job: bool = True,
) -> list[Path]:
    """Write each layer to a Gerber file and an optional Ucamco `.gbrjob`.

    Layout units default to um. File units default to mm, so a 100 um pad is
    0.1 mm on the board rather than 100 mm. Pass `layout_unit="mm"` if the
    Component is already in millimetres.

    Args:
        component: to export.
        dirpath: directory path.
        layermap_to_gerber_layer: map of GDS layer to GerberLayer.
        options: Gerber image options (`mode`, `layout_unit`, `resolution`).
        board: optional job-file size (mm) and copper layer count.
        write_job: write `<component>.gbrjob` next to the image files.

    Returns:
        Paths of files written, job file last when `write_job` is true.

    Example:
        import gdsfactory as gf
        from gdsfactory.export.to_gerber import GerberLayer, to_gerber
        from gdsfactory.gpdk import LAYER

        c = gf.components.pad()
        to_gerber(
            c,
            dirpath="gerber",
            layermap_to_gerber_layer={
                LAYER.MTOP: GerberLayer(
                    name="F_Cu",
                    function=["Copper", "L1", "Top"],
                    polarity="Positive",
                )
            },
        )
    """
    from gdsfactory.pdk import get_layer_tuple

    options = options or GerberOptions()
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    digits = decimal_digits(options.resolution)
    scale = file_unit_scale(options.layout_unit, options.mode)

    # Keys must match layermap tuples. Default `by="index"` uses klayout layer
    # indexes, so exported layers silently wrote empty files (#4748).
    layer_to_polygons = component.get_polygons_points(by="tuple")
    written: list[Path] = []
    job_files: list[tuple[str, GerberLayer]] = []

    names_seen: dict[str, str] = {}

    for layer_spec, layer in layermap_to_gerber_layer.items():
        layer_tup = get_layer_tuple(layer_spec)
        name = _gerber_filename(layer.name)
        if name in names_seen:
            raise ValueError(
                f"Gerber layers {names_seen[name]!r} and {layer.name!r} both map to "
                f"{name!r}. Layer names must produce distinct filenames."
            )
        names_seen[name] = layer.name
        filename = dirpath / name
        job_files.append((filename.name, layer))

        with open(filename, "w") as f:
            header = options.header or [
                "Gerber file generated by gdsfactory",
                f"Component: {component.name}",
            ]

            f.write("%TF.FileFunction," + ",".join(layer.function) + "*%\n")
            f.write(f"%TF.FilePolarity,{layer.polarity}*%\n")
            f.write(f"%TF.GenerationSoftware,gdsfactory,gdsfactory,{__version__}*%\n")
            f.write(format_specification(options.int_size, digits))
            f.writelines([f"G04 {line}*\n" for line in header])

            units = options.mode.upper()
            f.write(f"%MO{units}*%\n")
            f.write("%LPD*%\n")
            f.write("G01*\n")
            f.write("%ADD10C,0.050000*%\n")

            if layer_tup in layer_to_polygons:
                f.writelines(
                    polygon(
                        np.asarray(poly, dtype=float) * scale, decimal_digits=digits
                    )
                    for poly in layer_to_polygons[layer_tup]
                )

            f.write("M02*\n")
        written.append(filename)

    if write_job:
        job_path = dirpath / f"{component.name or 'gdsfactory'}.gbrjob"
        job_path.write_text(
            json.dumps(_job_file(component, job_files, options, board), indent=2) + "\n"
        )
        written.append(job_path)

    return written
