from __future__ import annotations

import shapely  # type: ignore
from trimesh.scene.scene import Scene

from gdsfactory.component import Component
from gdsfactory.technology import DerivedLayer, LayerStack, LayerViews, LogicalLayer
from gdsfactory.typings import LayerSpecs


def to_3d(
    component: Component,
    layer_views: LayerViews | None = None,
    layer_stack: LayerStack | None = None,
    exclude_layers: LayerSpecs | None = None,
) -> Scene:
    """Return Component 3D trimesh Scene.

    Args:
        component: to extrude in 3D.
        layer_views: layer colors from Klayout Layer Properties file.
            Defaults to active PDK.layer_views.
        layer_stack: contains thickness and zmin for each layer.
            Defaults to active PDK.layer_stack.
        exclude_layers: list of layer index to exclude.

    """
    from gdsfactory.pdk import (
        get_active_pdk,
        get_layer,
        get_layer_stack,
        get_layer_views,
    )

    try:
        from trimesh.creation import extrude_polygon  # type: ignore
        from trimesh.scene import Scene
    except ImportError as e:
        print("you need to `pip install trimesh`")
        raise e

    layer_views = layer_views or get_layer_views()
    layer_stack = layer_stack or get_layer_stack()

    scene = Scene()
    exclude_layers = exclude_layers or ()
    exclude_layers = [get_layer(layer) for layer in exclude_layers]

    component_with_booleans = layer_stack.get_component_with_derived_layers(component)
    polygons_per_layer = component_with_booleans.get_polygons_points(
        merge=True,
    )
    has_polygons = False

    for level in layer_stack.layers.values():
        layer = level.layer

        if isinstance(layer, LogicalLayer):
            layer_index = layer.layer
            layer_tuple: tuple[int, int] = tuple(layer_index)  # type: ignore

        elif isinstance(layer, DerivedLayer):
            assert level.derived_layer is not None
            layer_index = level.derived_layer.layer
            layer_tuple = tuple(layer_index)  # type: ignore
        else:
            raise ValueError(f"Layer {layer!r} is not a DerivedLayer or LogicalLayer")

        layer_index = int(get_layer(layer_index))  # type: ignore

        if layer_index in exclude_layers:
            continue

        if layer_index not in polygons_per_layer:
            continue

        zmin = level.zmin
        layer_view = layer_views.get_from_tuple(layer_tuple)
        assert layer_view.fill_color is not None
        color_rgb = [c / 255 for c in layer_view.fill_color.as_rgb_tuple(alpha=False)]
        if zmin is not None and layer_view.visible:
            has_polygons = True
            polygons = polygons_per_layer[layer_index]
            height = level.thickness
            for polygon in polygons:
                p = shapely.geometry.Polygon(polygon)
                mesh = extrude_polygon(p, height=height)
                mesh.apply_translation((0, 0, zmin))
                mesh.visual.face_colors = (*color_rgb, 0.5)
                scene.add_geometry(mesh)
    if not has_polygons:
        raise ValueError(
            f"{component.name!r} does not have polygons defined in the "
            f"layer_stack or layer_views for the active Pdk {get_active_pdk().name!r}"
        )
    return scene


if __name__ == "__main__":
    from gdsfactory.components import (
        grating_coupler_elliptical_trenches,
    )

    # c = gf.components.mzi()
    # c = gf.components.straight_heater_metal(length=40)
    # p = c.get_polygons_points()
    # c = gf.Component()
    # c << gf.c.rectangle(layer=(113, 0))

    c = grating_coupler_elliptical_trenches()
    # c = taper_strip_to_ridge_trenches()

    c.show()
    s = c.to_3d()
    s.show()  # type: ignore
