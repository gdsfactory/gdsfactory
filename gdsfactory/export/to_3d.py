from typing import Optional, Tuple

import shapely

from gdsfactory.component import Component
from gdsfactory.layers import LayerColors
from gdsfactory.pdk import get_layer_colors, get_layer_stack
from gdsfactory.tech import LayerStack
from gdsfactory.types import Layer


def to_3d(
    component: Component,
    layer_colors: Optional[LayerColors] = None,
    layer_stack: Optional[LayerStack] = None,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
):
    """Return Component 3D trimesh Scene.

    Args:
        component: to exture in 3D.
        layer_colors: layer colors from Klayout Layer Properties file.
            Defaults to active PDK.layer_colors.
        layer_stack: contains thickness and zmin for each layer.
            Defaults to active PDK.layer_stack.
        exclude_layers: layers to exclude.

    """
    try:
        import matplotlib.colors
        from trimesh.creation import extrude_polygon
        from trimesh.scene import Scene
    except ImportError as e:
        print("you need to `pip install trimesh`")
        raise e

    layer_colors = layer_colors or get_layer_colors()
    layer_stack = layer_stack or get_layer_stack()

    scene = Scene()
    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    exclude_layers = exclude_layers or ()

    has_polygons = False

    for layer, polygons in component.get_polygons(by_spec=True).items():
        if (
            layer not in exclude_layers
            and layer in layer_to_thickness
            and layer in layer_to_zmin
        ):
            height = layer_to_thickness[layer]
            zmin = layer_to_zmin[layer]
            layer_color = layer_colors.get_from_tuple(layer)
            color_hex = layer_color.color
            color_rgb = matplotlib.colors.to_rgb(color_hex)

            for polygon in polygons:
                p = shapely.geometry.Polygon(polygon)
                mesh = extrude_polygon(p, height=height)
                mesh.apply_translation((0, 0, zmin))
                mesh.visual.face_colors = (*color_rgb, 0.5)
                scene.add_geometry(mesh)
                has_polygons = True

    if not has_polygons:
        raise ValueError(
            f"{component.name!r} does not have polygons defined in the "
            "layer_stack or layer_colors for the active Pdk {get_active_pdk().name!r}"
        )
    return scene


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.taper_strip_to_ridge()
    # c = gf.components.straight()
    s = to_3d(c)
    s.show()
