"""Dummy fill to keep density constant."""
import itertools
from typing import Optional, Union

import gdspy
import numpy as np
from numpy import sqrt
from phidl.device_layout import _parse_layer
from phidl.geometry import (
    _expand_raster,
    _loop_over,
    _raster_index_to_coords,
    _rasterize_polygons,
)

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.types import Float2, Floats, LayerSpecs


@cell
def fill_cell_rectangle(
    size: Float2 = (20.0, 20.0),
    layers: LayerSpecs = (0, 1, 3),
    densities: Floats = (0.5, 0.25, 0.7),
    inverted=(False, False, False),
):
    """Creates a single Device on multiple layers to be used as fill

    based on phidl.geometry

    Args:
        size: array-like of int or float
            x, y dimensions of the fill area for all layers.
        layers: int, array-like[2], or set
            Specific layer(s) to put fill cell rectangle geometry on.
        densities: array-like of int or float
            Fill densities for each layer specified in ``layers``. Must be the same
            size as ``layers``.
        inverted: array-like or bool
            If true, inverts the fill area for corresponding layer. Must be the
            same size as ``layers``.

    """

    D = Component()
    for layer, density, inv in zip(layers, densities, inverted):
        rectangle_size = np.array(size) * sqrt(density)
        # r = D.add_ref(rectangle(size = rectangle_size, layer = layer))
        R = rectangle(size=tuple(rectangle_size), layer=layer, port_type=None)
        R.center = (0, 0)

        if inv is True:
            A = rectangle(size=size)
            A.center = (0, 0)
            A = A.get_polygons()
            B = R.get_polygons()
            p = gdspy.boolean(A, B, operation="not")
            D.add_polygon(p, layer=layer)
        else:
            D.add_ref(R)
    return D


@cell
def fill_rectangle(
    component: Component,
    fill_layers: LayerSpecs,
    fill_size: Float2 = (5.0, 5.0),
    avoid_layers: LayerSpecs = None,
    include_layers: LayerSpecs = None,
    margin: float = 5.0,
    fill_densities: Union[float, Floats] = (0.5, 0.25, 0.7),
    fill_inverted: bool = False,
    bbox: Optional[Float2] = None,
) -> Component:
    """Creates a rectangular fill pattern and fills all empty areas.

    in the input component and returns a component that contains just the fill
    Dummy fill keeps density constant during fabrication

    Args:
        component: Component to fill.
        fill_size: Rectangular size of the fill element.
        avoid_layers: Layers to be avoided (not filled) in D.
        include_layers: Layers to be filled, supercedes avoid_layers.
        margin:
            Margin spacing around avoided areas -- fill will not come within.
            `margin` of the geometry in D.
        fill_layers: list of layers. fill pattern layers.
        fill_densities: float between 0 and 1.
            Defines the fill pattern density (1.0 == fully filled).
        fill_inverted: Inverts the fill pattern.
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
            "[PHIDL] phidl.geometry.fill_rectangle() "
            "`fill_layers` and `fill_densities` parameters "
            "must be lists of the same length"
        )
    if len(fill_layers) != len(fill_inverted):
        raise ValueError(
            "[PHIDL] phidl.geometry.fill_rectangle() "
            "`fill_layers` and `fill_inverted` parameters must "
            "be lists of the same length"
        )

    fill_cell = fill_cell_rectangle(
        size=fill_size,
        layers=fill_layers,
        densities=fill_densities,
        inverted=fill_inverted,
    )
    F = Component()

    if avoid_layers == "all":
        exclude_polys = D.get_polygons(by_spec=False, depth=None)
    else:
        avoid_layers = [_parse_layer(layer) for layer in _loop_over(avoid_layers)]
        exclude_polys = D.get_polygons(by_spec=True, depth=None)
        exclude_polys = {
            key: exclude_polys[key] for key in exclude_polys if key in avoid_layers
        }
        exclude_polys = itertools.chain.from_iterable(exclude_polys.values())

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
                # F.add(gdspy.CellArray(ref_cell = fill_cell,
                #                       columns = len(s), rows = 1,
                #                       spacing = fill_size, ))
                a = F.add_array(fill_cell, columns=len(s), rows=1, spacing=fill_size)
                a.move((x, y))
            j += len(s)
    return F


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    c = gf.add_padding_container(c)
    c.unlock()
    c << fill_rectangle(
        c,
        fill_layers=((2, 0),),
        # fill_densities=(1.0,),
        fill_densities=0.5,
        avoid_layers=((1, 0),),
        # bbox=(100.0, 100.0),
    )
    c.show()
