from __future__ import annotations

from typing import Optional, Tuple

import shapely

from gdsfactory import generic_tech as generic
from gdsfactory.component import Component
from gdsfactory.technology import LayerStack, LayerViews
from gdsfactory.types import Layer


def to_3d(
    component: Component,
    layer_views: Optional[LayerViews] = generic.LAYER_VIEWS,
    layer_stack: Optional[LayerStack] = generic.LAYER_STACK,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
):
    """Return Component 3D trimesh Scene.

    Args:
        component: to extrude in 3D.
        layer_views: layer colors from Klayout Layer Properties file.
            Defaults to active PDK.layer_views.
        layer_stack: contains thickness and zmin for each layer.
            Defaults to active PDK.layer_stack.
        exclude_layers: layers to exclude.

    """
    from gdsfactory.pdk import get_layer_stack, get_layer_views

    try:
        from trimesh.creation import extrude_polygon
        from trimesh.scene import Scene
    except ImportError as e:
        print("you need to `pip install trimesh`")
        raise e

    layer_views = layer_views or get_layer_views()
    layer_stack = layer_stack or get_layer_stack()

    scene = Scene()
    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    exclude_layers = exclude_layers or ()

    has_polygons = False

    for layer, polygons in component.get_polygons(by_spec=True, as_array=False).items():
        if (
            layer not in exclude_layers
            and layer in layer_to_thickness
            and layer in layer_to_zmin
        ):
            height = layer_to_thickness[layer]
            zmin = layer_to_zmin[layer]
            layer_view = layer_views.get_from_tuple(layer)

            for polygon in polygons:
                p = shapely.geometry.Polygon(polygon.points)
                mesh = extrude_polygon(p, height=height)
                mesh.apply_translation((0, 0, zmin))
                mesh.visual.face_colors = (*layer_view.fill_color.as_rgb_tuple(), 0.5)
                scene.add_geometry(mesh)
                has_polygons = True

    if not has_polygons:
        raise ValueError(
            f"{component.name!r} does not have polygons defined in the "
            "layer_stack or layer_views for the active Pdk {get_active_pdk().name!r}"
        )
    return scene


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.taper_strip_to_ridge()
    # c = gf.components.straight()
    s = to_3d(c)
    s.show()
