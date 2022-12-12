from __future__ import annotations

from typing import Dict

import numpy as np

from gdsfactory.tech import LayerLevel, LayerStack


def bufferize(layerstack: LayerStack):
    """Convert layers without a z_to_bias to an equivalent z_to_bias.

    Arguments:
        layerstack: layerstack to process
    """
    for layer in layerstack.layers.values():
        if layer.z_to_bias is None:
            if layer.sidewall_angle is None:
                layer.z_to_bias = ([0, 1], [0, 0])
            else:
                buffer_magnitude = layer.thickness * np.tan(
                    np.radians(layer.sidewall_angle)
                )
                layer.z_to_bias = ([0, 1], [0, -1 * buffer_magnitude])

    return layerstack


def process_buffers(layer_polygons_dict: Dict, layerstack: LayerStack):
    """Break up layers into sub-layers according to z_to_bias.

    Arguments:
        layer_polygons_dict: dict of GDS layernames: shapely polygons
        layerstack: original Layerstack

    Returns:
        extended_layer_polygons_dict: dict of simulation_layername: (gds_layername, next_simulation_layername, this_layer_polygons, next_layer_polygons)
        extended_layerstack: LayerStack of simulation_layername: simulation layers
    """
    extended_layer_polygons_dict = {}
    extended_layerstack_layers = {}

    layerstack = bufferize(layerstack)

    for layername, polygons in layer_polygons_dict.items():
        zs = layerstack.layers[layername].z_to_bias[0]
        width_buffers = layerstack.layers[layername].z_to_bias[1]
        for ind, (z, width_buffer) in enumerate(zip(zs[:-1], width_buffers[:-1])):
            new_zmin = (
                layerstack.layers[layername].zmin
                + layerstack.layers[layername].thickness * z
            )
            new_thickness = (
                layerstack.layers[layername].thickness * zs[ind + 1]
                - layerstack.layers[layername].thickness * z
            )
            extended_layerstack_layers[f"{layername}_{z}"] = LayerLevel(
                thickness=new_thickness,
                zmin=new_zmin,
                material=layerstack.layers[layername].material,
                info=layerstack.layers[layername].info,
            )
            extended_layer_polygons_dict[f"{layername}_{z}"] = (
                f"{layername}",
                f"{layername}_{zs[ind+1]}",
                polygons.buffer(width_buffer),
                polygons.buffer(width_buffers[ind + 1]),
            )
        extended_layerstack_layers[f"{layername}_{zs[-1]}"] = LayerLevel(
            thickness=0,
            zmin=layerstack.layers[layername].zmin
            + layerstack.layers[layername].thickness,
            material=layerstack.layers[layername].material,
            info=layerstack.layers[layername].info,
        )

    return extended_layer_polygons_dict, LayerStack(layers=extended_layerstack_layers)
