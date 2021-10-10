from typing import Optional, Tuple

import matplotlib.colors
import shapely
from trimesh.creation import extrude_polygon
from trimesh.scene import Scene

from gdsfactory.component import Component
from gdsfactory.layers import LayerSet
from gdsfactory.tech import LAYER_STACK, LayerStack
from gdsfactory.types import Layer


def to_3d(
    component: Component,
    layer_set: LayerSet,
    layer_stack: LayerStack = LAYER_STACK,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
) -> Scene:
    """Return the Component 3D trimesh Scene.

    Args:
        component:
        layer_set: layer colors from Klayout Layer Properties file
        layer_stack: contains thickness and zmin for each layer
        exclude_layers: layers to exclude

    """
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
            for polygon in polygons:
                p = shapely.geometry.Polygon(polygon)
                mesh = extrude_polygon(p, height=height)
                mesh.apply_translation((0, 0, zmin))
                mesh.visual.face_colors = (*color_rgb, 0.5)
                scene.add_geometry(mesh)
    return scene


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.taper_strip_to_ridge()
    s = to_3d(c, layer_set=gf.layers.LAYER_SET)
    s.show()
