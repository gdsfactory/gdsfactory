"""Support for font rendering in GDS files.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

import freetype  # type: ignore
import numpy as np
import numpy.typing as npt
from matplotlib import font_manager
from matplotlib.path import Path

from gdsfactory.boolean import boolean
from gdsfactory.component import Component

_cached_fonts: dict[str, freetype.Face] = {}

try:
    import freetype  # type: ignore
except ImportError:
    print(
        "gdsfactory requires freetype to use real fonts. "
        "Either use the default DEPLOF font or install the freetype package:"
        "\n\n $ pip install freetype-py"
        "\n\n (Note: Windows users may have to find and replace the 'libfreetype.dll' "
        "file in their Python package directory /freetype/ with the correct one"
        "from here: https://github.com/ubawurinna/freetype-windows-binaries"
        " -- be sure to rename 'freetype.dll' to 'libfreetype.dll') "
    )


def _get_font_by_file(file: str) -> freetype.Face:
    """Load font file.

    Args:
        file: Load a font face from a given file.
    """
    # Cache opened fonts
    if file in _cached_fonts:
        return _cached_fonts[file]

    font_renderer = freetype.Face(file)
    font_renderer.set_char_size(32 * 64)  # 32pt size
    _cached_fonts[file] = font_renderer
    return font_renderer


def _get_font_by_name(name: str) -> freetype.Face:
    """Try to load a system font by name.

    Args:
        name: Load a system font.
    """
    try:
        font_file = font_manager.findfont(name, fallback_to_default=False)
    except Exception as e:
        raise ValueError(
            f"Failed to find font: {name!r}"
            "Try specifying the exact (full) path to the .ttf or .otf file. "
            "Otherwise, it might be resolved by rebuilding the matplotlib font cache"
        ) from e
    return _get_font_by_file(font_file)


def _get_glyph(font: freetype.Face, letter: str) -> tuple[Component, float, float]:
    """Get a block reference to the given letter."""
    if not isinstance(letter, str) and len(letter) == 1:
        raise TypeError(f"Letter must be a string of length 1. Got: {letter!r}")

    if not isinstance(font, freetype.Face):
        raise TypeError(
            f"font {font!r} must be a freetype font face. "
            "Load a font using _get_font_by_name first."
        )

    if getattr(font, "gds_glyphs", None) is None:
        font.gds_glyphs = {}

    if letter in font.gds_glyphs:
        return font.gds_glyphs[letter]  # type: ignore

    # Get the font name
    font_name = font.family_name.decode().replace(" ", "_")
    block_name = f"*char_{font_name}_0x{ord(letter):2X}"

    # Load control points from font file
    font.load_char(letter, freetype.FT_LOAD_FLAGS["FT_LOAD_NO_BITMAP"])
    glyph = font.glyph
    outline = glyph.outline
    points = np.array(outline.points, dtype=float) / font.size.ascender
    tags = outline.tags

    # Add polylines
    start, end = 0, -1
    VERTS, CODES = [], []
    # Iterate over each contour
    for i in range(len(outline.contours)):
        end = outline.contours[i]
        points = outline.points[start : end + 1]
        points.append(points[0])
        tags = outline.tags[start : end + 1]
        tags.append(tags[0])

        segments = [
            [
                points[0],
            ],
        ]
        for j in range(1, len(points)):
            segments[-1].append(points[j])
            if tags[j] & (1 << 0) and j < (len(points) - 1):
                segments.append(
                    [
                        points[j],
                    ]
                )
        verts = [
            points[0],
        ]
        codes = [
            Path.MOVETO,
        ]
        for segment in segments:
            if len(segment) == 2:
                verts.extend(segment[1:])
                codes.extend([Path.LINETO])
            elif len(segment) == 3:
                verts.extend(segment[1:])
                codes.extend([Path.CURVE3, Path.CURVE3])
            else:
                verts.append(segment[1])
                codes.append(Path.CURVE3)
                for i in range(1, len(segment) - 2):
                    A, B = segment[i], segment[i + 1]
                    C = ((A[0] + B[0]) / 2.0, (A[1] + B[1]) / 2.0)
                    verts.extend([C, B])
                    codes.extend([Path.CURVE3, Path.CURVE3])
                verts.append(segment[-1])
                codes.append(Path.CURVE3)
        VERTS.extend(verts)
        CODES.extend(codes)
        start = end + 1

    path = Path(VERTS, CODES)
    polygons = path.to_polygons(closed_only=True)
    # patch = PathPatch(path)

    # Construct the component
    component = Component()

    orientation = freetype.FT_Outline_Get_Orientation(outline._FT_Outline)
    polygons_cw = [p for p in polygons if _polygon_orientation(np.array(p)) == 0]
    polygons_ccw = [p for p in polygons if _polygon_orientation(np.array(p)) == 1]
    c1 = Component()
    c2 = Component()
    for p in polygons_cw:
        c1.add_polygon(np.array(p), layer=(1, 0))
    for p in polygons_ccw:
        c2.add_polygon(np.array(p), layer=(1, 0))
    if orientation == 0:
        # TrueType specification, fill the clockwise contour
        component = boolean(c1, c2, operation="not")
    elif orientation == 1:
        # PostScript specification, fill the counterclockwise contour
        component = boolean(c2, c1, operation="not")
    else:
        raise ValueError(f"Unknown orientation {orientation} for letter {letter}")

    component.name = block_name

    # Cache the return value and return it
    font.gds_glyphs[letter] = (component, glyph.advance.x, font.size.ascender)
    return font.gds_glyphs[letter]  # type: ignore


def _polygon_orientation(vertices: npt.NDArray[np.float64]) -> int:
    """Determine the orientation of a polygon."""
    n = len(vertices)
    if n < 3:
        raise ValueError("A polygon must have at least 3 vertices")
    s = 0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        s += (x2 - x1) * (y2 + y1)

    return 0 if s > 0 else 1


if __name__ == "__main__":
    from gdsfactory.components import text_freetype

    c = text_freetype("hello")
    c.show()
