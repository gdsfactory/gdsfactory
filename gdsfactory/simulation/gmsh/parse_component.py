"""Preprocessing involving both the GDS and the LayerStack, or the resulting simulation polygons."""
from __future__ import annotations

from typing import Dict

import numpy as np
from shapely.affinity import scale
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

from gdsfactory.simulation.gmsh.parse_gds import to_polygons
from gdsfactory.technology import LayerLevel, LayerStack


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
                layer.z_to_bias = (
                    (
                        [0, layer.width_to_z, 1],
                        [
                            1 * buffer_magnitude * layer.width_to_z,
                            0,
                            -1 * buffer_magnitude * (1 - layer.width_to_z),
                        ],
                    )
                    if layer.width_to_z
                    else ([0, 1], [0, -1 * buffer_magnitude])
                )
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
                layer=layerstack.layers[layername].layer,
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
            layer=layerstack.layers[layername].layer,
            thickness=0,
            zmin=layerstack.layers[layername].zmin
            + layerstack.layers[layername].thickness,
            material=layerstack.layers[layername].material,
            info=layerstack.layers[layername].info,
        )

    return extended_layer_polygons_dict, LayerStack(layers=extended_layerstack_layers)


def buffers_to_lists(layer_polygons_dict: Dict, layerstack: LayerStack):
    """Break up polygons on each layer into lists of polygons:z tuples according to z_to_bias.

    Arguments:
        layer_polygons_dict: dict of GDS layernames: shapely polygons
        layerstack: original Layerstack

    Returns:
        extended_layer_polygons_dict: dict of layername: List[(z, polygon_at_z)] for all polygons at z
    """
    extended_layer_polygons_dict = {}

    layerstack = bufferize(layerstack)

    xfactor, yfactor = 1, 1  # buffer_to_scaling(polygon, width_buffer)
    for layername, polygons in layer_polygons_dict.items():
        all_polygons_list = []
        for polygon in polygons.geoms if hasattr(polygons, "geoms") else [polygons]:
            zs = layerstack.layers[layername].z_to_bias[0]
            width_buffers = layerstack.layers[layername].z_to_bias[1]

            polygons_list = [
                (
                    z
                    * (
                        layerstack.layers[layername].zmin
                        + layerstack.layers[layername].thickness
                    )
                    + layerstack.layers[layername].zmin,
                    scale(polygon, xfact=xfactor, yfact=yfactor),
                )
                for z, width_buffer in zip(zs, width_buffers)
            ]
            all_polygons_list.append(polygons_list)
        extended_layer_polygons_dict[layername] = all_polygons_list

    return extended_layer_polygons_dict


def merge_by_material_func(layer_polygons_dict: Dict, layerstack: LayerStack):
    """Merge polygons of layer_polygons_dict whose layerstack keys share the same material in layerstack values.

    Returns new layer_polygons_dict with merged polygons and materials as keys.
    """
    merged_layer_polygons_dict = {}
    for layername, polygons in layer_polygons_dict.items():
        material = layerstack.layers[layername].material
        if material in merged_layer_polygons_dict:
            merged_layer_polygons_dict[material] = unary_union(
                MultiPolygon(
                    to_polygons([merged_layer_polygons_dict[material], polygons])
                )
            )
        else:
            merged_layer_polygons_dict[material] = polygons

    return merged_layer_polygons_dict


def create_2D_surface_interface(
    layer_polygons: MultiPolygon,
    thickness_min: float = 0.0,
    thickness_max: float = 0.01,
    simplify: float = 0.005,
):
    """Create 2D entity at the interface of two layers/materials.

    Arguments:
        layer_polygons: shapely polygons.
        thickness_min: distance to define the interfacial region towards the polygon.
        thickness_max: distance to define the interfacial region away from the polygon.
        simplify: simplification factor for over-parametrized geometries

    Returns:
        shapely interface polygon
    """
    interfaces = layer_polygons.boundary
    interface_surface = layer_polygons.boundary
    left_hand_side = interfaces.buffer(thickness_max, single_sided=True)
    right_hand_side = interfaces.buffer(-thickness_min, single_sided=True)
    interface_surface = left_hand_side.union(right_hand_side)

    return interface_surface.simplify(simplify, preserve_topology=False)
