from __future__ import annotations

import pathlib

from gdsfactory.component import Component
from gdsfactory.technology import DerivedLayer, LayerStack, LogicalLayer
from gdsfactory.typings import LayerSpecs, PathType


def to_stl(
    component: Component,
    filepath: PathType,
    layer_stack: LayerStack | None = None,
    exclude_layers: LayerSpecs | None = None,
    hull_invalid_polygons: bool = False,
    scale: float | None = None,
) -> None:
    """Exports a Component into STL.

    Args:
        component: to export.
        filepath: filepath prefix to write STL to.
            Each file will have each exported layer as suffix.
        layer_stack: contains thickness and zmin for each layer.
        exclude_layers: list of layer index to exclude.
        hull_invalid_polygons: If True, replaces invalid polygons (determined by shapely.Polygon.is_valid) with its convex hull.
        scale: Optional factor by which to scale meshes before writing.

    """
    import shapely
    import trimesh.creation

    from gdsfactory.pdk import get_active_pdk, get_layer, get_layer_stack

    layer_stack = layer_stack or get_layer_stack()
    has_polygons = False

    filepath = pathlib.Path(filepath)
    exclude_layers = exclude_layers or ()
    exclude_layers = [get_layer(layer) for layer in exclude_layers]

    component_with_booleans = layer_stack.get_component_with_derived_layers(component)
    polygons_per_layer = component_with_booleans.get_polygons_points()

    for level in layer_stack.layers.values():
        layer = level.layer

        if isinstance(layer, LogicalLayer):
            layer_index = layer.layer

        elif isinstance(layer, DerivedLayer):
            assert level.derived_layer is not None
            layer_index = level.derived_layer.layer
        else:
            raise ValueError(f"Layer {layer!r} is not a DerivedLayer or LogicalLayer")

        layer_tuple: tuple[int, int] = tuple(layer_index)  # type: ignore

        if layer_index in exclude_layers:
            continue

        if layer_index not in polygons_per_layer:
            continue

        zmin = level.zmin
        if zmin is not None:  # type: ignore
            has_polygons = True
            polygons = polygons_per_layer[layer_index]
            height = level.thickness
            layer_name = level.name or f"{layer_tuple[0]}_{layer_tuple[1]}"
            filepath_layer = (
                filepath.parent / f"{filepath.stem}_{layer_name}{filepath.suffix}"
            )
            print(
                f"Write {filepath_layer.absolute()!r} zmin = {zmin:.3f}, height = {height:.3f}"
            )
            meshes: list[trimesh.Trimesh] = []
            for polygon in polygons:
                p = shapely.geometry.Polygon(polygon)

                if hull_invalid_polygons and not p.is_valid:
                    p = p.convex_hull

                mesh = trimesh.creation.extrude_polygon(p, height=height)  # type: ignore
                mesh.apply_translation((0, 0, zmin))
                meshes.append(mesh)

        layer_mesh = trimesh.util.concatenate(meshes)  # type: ignore

        if scale:
            layer_mesh.apply_scale(scale)  # type: ignore

        layer_mesh.export(filepath_layer)  # type: ignore

    if not has_polygons:
        raise ValueError(
            f"{component.name!r} does not have polygons defined in the "
            f"layer_stack or layer_views for the active Pdk {get_active_pdk().name!r}"
        )


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.grating_coupler_elliptical_trenches()  # type: ignore
    c.show()
    to_stl(c, filepath="a.stl")
