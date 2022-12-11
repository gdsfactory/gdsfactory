from __future__ import annotations

import numpy as np

from gdsfactory.tech import LayerStack, LayerLevel
from typing import Dict


def bufferize(layerstack: LayerStack):
    """Convert layers without a buffer_profile to an equivalent buffer_profile.

    Arguments:
        layerstack: layerstack to process
    """
    for layername, layer in layerstack.layers.items():
        if layer.buffer_profile is None:
            if layer.sidewall_angle is None:
                layer.buffer_profile = ([0,1], [0, 0])
            else:
                buffer_magnitude = layer.thickness * np.tan(np.radians(layer.sidewall_angle))
                layer.buffer_profile = ([0,1], [0, -1*buffer_magnitude])

    return layerstack


def process_buffers(layer_polygons_dict: Dict, layerstack: LayerStack):
    """Break up layers into sub-layers according to buffer_profile.

    Arguments:
        layer_polygons_dict: dict of GDS layernames: shapely polygons
        layerstack: original Layerstack

    Returns:
        extended_layer_polygons_dict: dict of (GDS layername, simulation_layername, next_simulation_layername): polygons
        extended_layerstack: LayerStack of simulation layers
    """
    extended_layer_polygons_dict = {}
    extended_layerstack_layers = {}

    for layername, layer in layerstack.layers.items():
        print(layername, layer)

    layerstack = bufferize(layerstack)
    print("=========================================")

    for layername, layer in layerstack.layers.items():
        print(layername, layer)

    for layername, polygons in layer_polygons_dict.items():
        zs = layerstack.layers[layername].buffer_profile[0]
        width_buffers = layerstack.layers[layername].buffer_profile[1]
        for ind, (z, width_buffer) in enumerate(zip(zs[:-1], width_buffers[:-1])):
            new_zmin = layerstack.layers[layername].zmin + layerstack.layers[layername].thickness * z
            new_thickness = layerstack.layers[layername].thickness * zs[ind+1] - layerstack.layers[layername].thickness * z
            extended_layerstack_layers[f"{layername}_{z}"] = LayerLevel(thickness=new_thickness, zmin=new_zmin, material=layerstack.layers[layername].material, info=layerstack.layers[layername].info,)
            extended_layer_polygons_dict[f"{layername}_{z}"] = polygons.buffer(width_buffer)

    return extended_layer_polygons_dict, LayerStack(layers=extended_layerstack_layers)