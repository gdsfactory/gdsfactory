import meow as mw

from gdsfactory.tech import LAYER, LayerStack

"""Conversion between gdsfactory material names and meow materials class."""
gdsfactory_to_meow_materials = {
    "si": mw.silicon,
    "sio2": mw.silicon_oxide,
}


def add_global_layers(component, layerstack):
    """Adds bbox polygons for global layers."""
    bbox = component.bbox
    for layername, layer in layerstack.layers.items():
        if layer["layer"] == LAYER.WAFER:
            component.add_ref(
                gf.components.box(bbox[0, 0], bbox[0, 1], bbox[1, 0], bbox[1, 1])
            )
        else:
            continue
    return component


def layerstack_to_extrusion(layerstack: LayerStack):
    """Convert LayerStack to meow extrusions."""
    extrusions = {}
    for layername, layer in layerstack.layers.items():
        if layer.layer not in extrusions.keys():
            extrusions[layer.layer] = []
        extrusions[layer.layer].append(
            mw.GdsExtrusionRule(
                material=gdsfactory_to_meow_materials[layer.material],
                h_min=layer.zmin,
                h_max=layer.zmin + layer.thickness,
                mesh_order=layer.info["mesh_order"],
            )
        )
    return extrusions


if __name__ == "__main__":

    import gdsfactory as gf

    c = gf.components.taper_cross_section_linear()
    c.show()

    from gdsfactory.tech import get_layer_stack_generic

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
                "box",
                "clad",
            )
        }
    )

    print(layerstack_to_extrusion(filtered_layerstack))
