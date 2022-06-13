from typing import Optional, Tuple

import matplotlib.colors
import shapely

from gdsfactory.component import Component
from gdsfactory.layers import LayerSet
from gdsfactory.pdk import get_layer_set, get_layer_stack
from gdsfactory.tech import LayerStack
from gdsfactory.types import Layer


def to_3d(
    component: Component,
    layer_set: Optional[LayerSet] = None,
    layer_stack: Optional[LayerStack] = None,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
):
    """Return Component 3D trimesh Scene.

    Args:
        component: to exture in 3D.
        layer_set: layer colors from Klayout Layer Properties file.
            Defaults to active PDK.layer_set.
        layer_stack: contains thickness and zmin for each layer.
            Defaults to active PDK.layer_stack.
        exclude_layers: layers to exclude.

    """
    try:
        from trimesh.creation import extrude_polygon
        from trimesh.scene import Scene
    except ImportError:
        print("you need to `pip install trimesh`")

    layer_set = layer_set or get_layer_set()
    layer_stack = layer_stack or get_layer_stack()

    scene = Scene()
    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    exclude_layers = exclude_layers or []

    for layer, polygons in component.get_polygons(by_spec=True).items():
        if (
            layer not in exclude_layers
            and layer in layer_to_thickness
            and layer in layer_to_zmin
        ):
            height = layer_to_thickness[layer]
            zmin = layer_to_zmin[layer]
            color_hex = layer_set.get_from_tuple(layer).color
            color_rgb = matplotlib.colors.to_rgb(color_hex)

            print(layer, color_hex, color_rgb, height, zmin)

            for polygon in polygons:
                p = shapely.geometry.Polygon(polygon)
                mesh = extrude_polygon(p, height=height)
                mesh.apply_translation((0, 0, zmin))
                mesh.visual.face_colors = (*color_rgb, 0.5)
                scene.add_geometry(mesh)
    return scene


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.components.taper_strip_to_ridge()
    c = gf.components.straight()
    s = to_3d(c)
    s.show()
