from __future__ import annotations

import pathlib
from typing import Optional, Tuple

import gdsfactory.generic_tech as generic
from gdsfactory.component import Component
from gdsfactory.technology import LayerStack, LayerViews
from gdsfactory.typings import Layer


def to_stl(
    component: Component,
    filepath: str,
    layer_views: LayerViews = generic.LAYER_VIEWS,
    layer_stack: LayerStack = generic.LAYER_STACK,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
) -> None:
    """Exports a Component into STL.

    Args:
        component: to export.
        filepath: to write STL to.
        layer_views: layer colors from Klayout Layer Properties file.
        layer_stack: contains thickness and zmin for each layer.
        exclude_layers: layers to exclude.

    """
    import shapely
    from trimesh.creation import extrude_polygon

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    filepath = pathlib.Path(filepath)
    exclude_layers = exclude_layers or []

    for layer, polygons in component.get_polygons(by_spec=True).items():
        if (
            layer not in exclude_layers
            and layer in layer_to_thickness
            and layer in layer_to_zmin
        ):
            height = layer_to_thickness[layer]
            zmin = layer_to_zmin[layer]
            filepath_layer = (
                filepath.parent
                / f"{filepath.stem}_{layer[0]}_{layer[1]}{filepath.suffix}"
            )
            for polygon in polygons:
                p = shapely.geometry.Polygon(polygon)
                mesh = extrude_polygon(p, height=height)
                mesh.apply_translation((0, 0, zmin))
                mesh.visual.face_colors = (
                    *layer_views.get_from_tuple(layer).fill_color.as_rgb_tuple(),
                    0.5,
                )
                mesh.export(filepath_layer)


if __name__ == "__main__":
    import gdsfactory as gf
    import gdsfactory.generic_tech as generic

    c = gf.components.taper_strip_to_ridge()
    to_stl(c, layer_views=generic.LAYER_VIEWS, filepath="a.stl")
