from typing import Optional, Tuple

import matplotlib.colors
import shapely
from trimesh.creation import extrude_polygon
from trimesh.scene import Scene

from gdsfactory.component import Component
from gdsfactory.layers import LayerSet
from gdsfactory.tech import LAYER_STACK, LayerStack
from gdsfactory.types import Layer


def to_trimesh(
    component: Component,
    layer_set: LayerSet,
    layer_stack: LayerStack = LAYER_STACK,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
) -> Scene:
    """Return 3D trimesh scene for a component.

    Args:
        component:
        layer_set: contains color from Klayout Layer Properties file
        layer_stack: contains z, zmin for each layer
        exclude_process_op: Exclude all process ops in this list.

    """
    scene = Scene()
    layer_to_z = layer_stack.get_layer_to_thickness_nm()
    layer_to_zmin = layer_stack.get_layer_to_zmin_nm()
    exclude_layers = exclude_layers or []

    for layer, polygons in component.get_polygons(by_spec=True).items():
        if (
            layer not in exclude_layers
            and layer in layer_to_z
            and layer in layer_to_zmin
        ):
            height = layer_to_z[layer] * 1e-3
            zmin = layer_to_zmin[layer] * 1e-3
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
    scene = to_trimesh(c, layer_set=gf.lys)
    scene.show()
