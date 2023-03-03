from __future__ import annotations

import pathlib
from typing import Optional, Tuple

from gdsfactory.component import Component
from gdsfactory.technology import LayerStack, LayerViews
from gdsfactory.typings import Layer


def to_stl(
    component: Component,
    filepath: str,
    layer_views: Optional[LayerViews] = None,
    layer_stack: Optional[LayerStack] = None,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
    hull_invalid_polygons: bool = True,
    scale: Optional[float] = None,
) -> None:
    """Exports a Component into STL.

    Args:
        component: to export.
        filepath: to write STL to.
        layer_views: layer colors from Klayout Layer Properties file.
        layer_stack: contains thickness and zmin for each layer.
        exclude_layers: layers to exclude.
        hull_invalid_polygons: If True, replaces invalid polygons (determined by shapely.Polygon.is_valid) with its convex hull.
        scale: Optional factor by which to scale meshes before writing.

    """
    import shapely
    import trimesh
    from trimesh.creation import extrude_polygon
    from gdsfactory.pdk import get_layer_stack

    layer_stack = layer_stack or get_layer_stack()

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    filepath = pathlib.Path(filepath)
    exclude_layers = exclude_layers or []

    component_with_booleans = layer_stack.get_component_with_derived_layers(component)
    component_layers = component_with_booleans.get_layers()

    for layer, polygons in component_with_booleans.get_polygons(by_spec=True).items():
        if (
            layer not in exclude_layers
            and layer in layer_to_thickness
            and layer in layer_to_zmin
            and layer in component_layers
        ):
            height = layer_to_thickness[layer]
            zmin = layer_to_zmin[layer]
            filepath_layer = (
                filepath.parent
                / f"{filepath.stem}_{layer[0]}_{layer[1]}{filepath.suffix}"
            )
            print(f"Write {filepath_layer.absolute()!r}")
            meshes = []
            for polygon in polygons:
                p = shapely.geometry.Polygon(polygon)

                if hull_invalid_polygons and not p.is_valid:
                    p = p.convex_hull

                mesh = extrude_polygon(p, height=height)
                mesh.apply_translation((0, 0, zmin))
                meshes.append(mesh)

            layer_mesh = trimesh.util.concatenate(meshes)

            if scale:
                layer_mesh.apply_scale(scale)

            layer_mesh.export(filepath_layer)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.taper_strip_to_ridge()
    c.show()
    to_stl(c, filepath="a.stl")
