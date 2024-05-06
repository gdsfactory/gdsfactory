from __future__ import annotations

import pathlib

from gdsfactory.component import Component
from gdsfactory.technology import LayerStack, LayerViews
from gdsfactory.typings import Layer


def to_stl(
    component: Component,
    filepath: str,
    layer_views: LayerViews | None = None,
    layer_stack: LayerStack | None = None,
    exclude_layers: tuple[Layer, ...] | None = None,
    use_layer_name: bool = False,
    hull_invalid_polygons: bool = False,
    scale: float | None = None,
) -> None:
    """Exports a Component into STL.

    Args:
        component: to export.
        filepath: filepath prefix to write STL to.
            Each file will have each exported layer as suffix.
        layer_views: contains visibility information for each layer.
        layer_stack: contains thickness and zmin for each layer.
        exclude_layers: layers to exclude.
        use_layer_name: If True, uses LayerLevel names in output filenames rather than gds_layer and gds_datatype.
        hull_invalid_polygons: If True, replaces invalid polygons (determined by shapely.Polygon.is_valid) with its convex hull.
        scale: Optional factor by which to scale meshes before writing.

    """
    import shapely
    import trimesh.creation

    from gdsfactory.pdk import get_layer_stack, get_layer_views

    layer_stack = layer_stack or get_layer_stack()
    layer_views = layer_views or get_layer_views()

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    filepath = pathlib.Path(filepath)
    exclude_layers = exclude_layers or []

    component_with_booleans = layer_stack.get_component_with_derived_layers(component)

    component_layers = component_with_booleans.get_polygons().keys()
    layer_names = list(layer_stack.layers.keys())
    layer_tuples = list(layer_stack.layers.values())

    for layer, polygons in component_with_booleans.get_polygons().items():
        if (
            layer in exclude_layers
            or layer not in layer_to_thickness
            or layer not in layer_to_zmin
            or layer not in component_layers
        ):
            continue

        height = layer_to_thickness[layer]
        zmin = layer_to_zmin[layer]

        layer_name = (
            layer_names[layer_tuples.index(layer)]
            if use_layer_name
            else f"{layer[0]}_{layer[1]}"
        )

        filepath_layer = (
            filepath.parent / f"{filepath.stem}_{layer_name}{filepath.suffix}"
        )
        print(
            f"Write {filepath_layer.absolute()!r} zmin = {zmin:.3f}, height = {height:.3f}"
        )
        meshes = []
        layer_view = layer_views.get_from_tuple(layer)
        if zmin is not None and layer_view.visible:
            for polygon in [polygons]:
                p = shapely.geometry.Polygon(polygon)

                if hull_invalid_polygons and not p.is_valid:
                    p = p.convex_hull

                mesh = trimesh.creation.extrude_polygon(p, height=height)
                mesh.apply_translation((0, 0, zmin))
                meshes.append(mesh)

        layer_mesh = trimesh.util.concatenate(meshes)

        if scale:
            layer_mesh.apply_scale(scale)

        layer_mesh.export(filepath_layer)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.mzi()
    c.show()
    to_stl(c, filepath="a.stl")
