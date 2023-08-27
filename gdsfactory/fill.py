"""Dummy fill to keep density constant.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""
from __future__ import annotations

import itertools

import gdstk
import numpy as np
from numpy import sqrt

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.component_layout import _parse_layer
from gdsfactory.components.rectangle import rectangle
from gdsfactory.typings import (
    ComponentOrReference,
    ComponentSpec,
    Float2,
    Floats,
    LayerSpecs,
)


def _loop_over(var):
    """Checks if a variable is in the form of an iterable (list/tuple)\
    and if not, returns it as a list.

    Useful for allowing argument inputs to be either lists
    (e.g. [1, 3, 4]) or single-valued (e.g. 3).
    Returns list Variable converted to list if single-valued input.

    Args:
        var : int or float or list
            Variable to check for iterability.
    """
    return var if hasattr(var, "__iter__") else [var]


def _rasterize_polygons(
    polygons, bounds=([-100, -100], [100, 100]), dx: float = 1.0, dy: float = 1.0
):
    """Converts polygons to a black/white (1/0) matrix."""
    try:
        from skimage import draw
    except ImportError as e:
        raise ImportError(
            "The fill function requires the module "
            '"scikit-image" to operate.  Please retry '
            "after installing scikit-image:\n\n"
            "$ pip install --upgrade scikit-image"
        ) from e

    # Initialize the raster matrix we'll be writing to
    xsize = int((bounds[1][0] - bounds[0][0]) // dx)
    ysize = int((bounds[1][1] - bounds[0][1]) // dy)
    raster = np.zeros((ysize, xsize), dtype=bool)

    # Prepare polygon array by shifting all points into the first quadrant and
    # separating points into x and y lists
    xpts = []
    ypts = []
    for p in polygons:
        p_array = np.asarray(p)
        x = p_array[:, 0]
        y = p_array[:, 1]
        xpts.append((x - (bounds[1][0] + bounds[0][0] - (xsize - 1) * dx) / 2) / dx)
        ypts.append((y - (bounds[1][1] + bounds[0][1] - (ysize - 1) * dy) / 2) / dy)

    # TODO: Replace polygon_perimeter with the supercover version
    for n in range(len(xpts)):
        rr, cc = draw.polygon(ypts[n], xpts[n], shape=raster.shape)
        rrp, ccp = draw.polygon_perimeter(
            ypts[n], xpts[n], shape=raster.shape, clip=False
        )
        raster[rr, cc] = 1
        raster[rrp, ccp] = 1

    return raster


def _raster_index_to_coords(
    i, j, bounds=([-100, -100], [100, 100]), dx: float = 1, dy: float = 1
):
    """Converts (i,j) index of raster matrix to real coordinates."""
    xsize = int((bounds[1][0] - bounds[0][0]) // dx)
    ysize = int((bounds[1][1] - bounds[0][1]) // dy)
    x = j * dx + (bounds[1][0] + bounds[0][0] - (xsize - 1) * dx) / 2
    y = i * dy + (bounds[1][1] + bounds[0][1] - (ysize - 1) * dy) / 2
    return x, y


def _expand_raster(raster, distance: tuple[float, float] = (4, 2)):
    """Expands all black (1) pixels in the raster."""
    try:
        from skimage import draw, morphology
    except ImportError as e:
        raise ImportError(
            "The fill function requires the module "
            '"scikit-image" to operate.  Please retry '
            "after installing scikit-image:\n\n"
            "$ pip install --upgrade scikit-image"
        ) from e
    if distance[0] <= 0.5 and distance[1] <= 0.5:
        return raster

    num_pixels = np.array(np.ceil(distance), dtype=int)
    neighborhood = np.zeros((num_pixels[1] * 2 + 1, num_pixels[0] * 2 + 1), dtype=bool)
    rr, cc = draw.ellipse(
        num_pixels[1], num_pixels[0], distance[1] + 0.5, distance[0] + 0.5
    )
    neighborhood[rr, cc] = 1

    return morphology.binary_dilation(image=raster, footprint=neighborhood)


@cell
def fill_cell_rectangle(
    size: Float2 = (20.0, 20.0),
    layers: LayerSpecs = (0, 1, 3),
    densities: Floats = (0.5, 0.25, 0.7),
    inverted=(False, False, False),
) -> Component:
    """Returns Component on multiple layers to be used as fill.

    based on phidl.geometry

    Args:
        size: x, y dimensions of the fill area for all layers.
        layers: Specific layer(s) to put fill cell rectangle geometry on.
        densities: Fill densities for each layer specified in ``layers``.
            Must be the same size as ``layers``.
        inverted: array-like or bool
            If true, inverts the fill area for corresponding layer.
            Must be the same size as ``layers``.

    """
    D = Component()
    for layer, density, inv in zip(layers, densities, inverted):
        rectangle_size = np.array(size) * sqrt(density)
        # r = D.add_ref(rectangle(size = rectangle_size, layer = layer))
        R = rectangle(
            size=tuple(rectangle_size), layer=layer, port_type=None, centered=True
        )

        if inv is True:
            A = rectangle(size=size, centered=True)
            A = A.get_polygons()
            B = R.get_polygons()
            p = gdstk.boolean(A, B, operation="not")
            D.add_polygon(p, layer=layer)
        else:
            D.add_ref(R)
    return D


@cell
def fill_rectangle(
    component: ComponentOrReference,
    fill_layers: LayerSpecs,
    fill_size=(5.0, 5.0),
    avoid_layers: LayerSpecs | None = None,
    include_layers: LayerSpecs | None = None,
    margin: float = 5.0,
    fill_densities: float | Floats = (0.5, 0.25, 0.7),
    fill_inverted: list[float] | None = None,
    bbox: object | None = None,
) -> Component:
    """Returns rectangular fill pattern and fills all empty areas.

    In the input component and returns a component that contains just the fill
    Dummy fill keeps density constant during fabrication

    Args:
        component: Component to fill.
        fill_layers: list of layers. fill pattern layers.
        fill_size: Rectangular size of the fill element.
        avoid_layers: Layers to be avoided (not filled) in D.
        include_layers: Layers to be filled, supersedes avoid_layers.
        margin: Margin spacing around avoided areas.
        fill_densities: defines the fill pattern density (1.0 == fully filled).
        fill_inverted: inverts the fill pattern.
        bbox: x, y limit the fill pattern to the area defined by this bounding box.

    """
    D = component

    # Create the fill cell.
    # If fill_inverted is not specified, assume all False
    fill_layers = _loop_over(fill_layers)
    fill_densities = _loop_over(fill_densities)
    if fill_inverted is None:
        fill_inverted = [False] * len(fill_layers)

    fill_inverted = _loop_over(fill_inverted)
    if len(fill_layers) != len(fill_densities):
        raise ValueError(
            "fill_rectangle() `fill_layers` and `fill_densities` parameters "
            "must be lists of the same length"
        )
    if len(fill_layers) != len(fill_inverted):
        raise ValueError(
            "fill_rectangle() `fill_layers` and `fill_inverted` parameters must "
            "be lists of the same length"
        )

    fill_cell = fill_cell_rectangle(
        size=fill_size,
        layers=fill_layers,
        densities=fill_densities,
        inverted=fill_inverted,
    )

    fill_cell = gf.get_component(fill_cell)
    F = Component()

    avoid_layers = [_parse_layer(layer) for layer in _loop_over(avoid_layers)]
    exclude_polys = D.get_polygons(by_spec=True, depth=None, as_array=False)

    if avoid_layers:
        exclude_polys = {
            key: exclude_polys[key] for key in exclude_polys if key in avoid_layers
        }

    exclude_polys = [
        polygon.points for polygons in exclude_polys.values() for polygon in polygons
    ]

    if include_layers is None:
        include_polys = []
    else:
        include_layers = [_parse_layer(layer) for layer in _loop_over(include_layers)]
        include_polys = D.get_polygons(by_spec=True, depth=None)
        include_polys = {
            key: include_polys[key] for key in include_polys if key in include_layers
        }
        include_polys = itertools.chain.from_iterable(include_polys.values())

    if bbox is None:
        bbox = D.bbox

    raster = _rasterize_polygons(
        polygons=exclude_polys, bounds=bbox, dx=fill_size[0], dy=fill_size[1]
    )
    raster = raster & ~_rasterize_polygons(
        polygons=include_polys, bounds=bbox, dx=fill_size[0], dy=fill_size[1]
    )
    raster = _expand_raster(raster, distance=margin / np.array(fill_size))

    for i in range(np.size(raster, 0)):
        sub_rasters = [list(g) for k, g in itertools.groupby(raster[i])]
        j = 0
        for s in sub_rasters:
            if s[0] == 0:
                x, y = _raster_index_to_coords(i, j, bbox, fill_size[0], fill_size[1])
                a = F.add_array(fill_cell, columns=len(s), rows=1, spacing=fill_size)
                a.move((x, y))
            j += len(s)
    return F


@cell
def fill_rectangle_custom(
    component: Component,
    fill_cell: ComponentSpec,
    spacing: tuple[float, float],
    avoid_layers: LayerSpecs | None = None,
    margin: float = 5.0,
    bbox: object | None = None,
) -> Component:
    """Returns custom fill pattern to fill all empty areas.

    In the input component and returns a component that contains just the fill
    Dummy fill keeps density constant during fabrication.

    Args:
        component: Component to fill.
        fill_cell: Component to use as fill cell.
        spacing: x, y pitch for fill.
        avoid_layers: Layers to be avoided (not filled) in D.
        margin: Margin spacing around avoided areas.
        bbox: x, y limit the fill pattern to the area defined by this bounding box.
    """
    D = component
    if bbox is None:
        bbox = D.bbox

    fill_cell = gf.get_component(fill_cell)
    F = Component()
    avoid_layers = [_parse_layer(layer) for layer in _loop_over(avoid_layers)]
    exclude_polys = D.get_polygons(by_spec=True, depth=None, as_array=False)

    if avoid_layers:
        exclude_polys = {
            key: exclude_polys[key] for key in exclude_polys if key in avoid_layers
        }

    exclude_polys = [
        polygon.points for polygons in exclude_polys.values() for polygon in polygons
    ]

    include_polys = []

    raster = _rasterize_polygons(
        polygons=exclude_polys, bounds=bbox, dx=spacing[0], dy=spacing[1]
    )
    raster = raster & ~_rasterize_polygons(
        polygons=include_polys, bounds=bbox, dx=spacing[0], dy=spacing[1]
    )
    raster = _expand_raster(raster, distance=margin / np.array(spacing))

    for i in range(np.size(raster, 0)):
        sub_rasters = [list(g) for k, g in itertools.groupby(raster[i])]
        j = 0
        for s in sub_rasters:
            if s[0] == 0:
                x, y = _raster_index_to_coords(i, j, bbox, spacing[0], spacing[1])
                a = F.add_array(fill_cell, columns=len(s), rows=1, spacing=spacing)
                a.move((x, y))
            j += len(s)
    return F


def test_fill() -> None:
    from gdsfactory.difftest import difftest

    c = gf.Component("test_fill")
    wg = c << gf.components.straight()
    _ = c << gf.add_padding_container(wg.parent, default=15)
    fill = fill_rectangle(
        c,
        fill_layers=((2, 0),),
        fill_densities=(1.0,),
        # fill_densities=0.5,
        avoid_layers=((1, 0),),
        # bbox=(100.0, 100.0),
    )
    _ = c << fill
    difftest(c, test_name="fill")


if __name__ == "__main__":
    test_fill()

    # c = gf.Component("test_fill")
    # wg = c << gf.components.straight()
    # _ = c << gf.add_padding_container(wg.parent, default=15)
    # fill = fill_rectangle(
    #     c,
    #     fill_layers=((2, 0),),
    #     fill_densities=(1.0,),
    #     # fill_densities=0.5,
    #     avoid_layers=((1, 0),),
    #     # bbox=(100.0, 100.0),
    # )
    # _ = c << fill
    # c.show()
