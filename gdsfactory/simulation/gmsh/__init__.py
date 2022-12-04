from __future__ import annotations

from gdsfactory.simulation.gmsh.mesh import create_physical_mesh, mesh_from_polygons
from gdsfactory.simulation.gmsh.meshtracker import MeshTracker
from gdsfactory.simulation.gmsh.parse_gds import (
    cleanup_component,
    fuse_polygons,
    round_coordinates,
    tile_shapes,
    to_polygons,
)
from gdsfactory.simulation.gmsh.parse_layerstack import (
    get_layer_overlaps_z,
    get_layers_at_z,
    list_unique_layerstack_z,
    map_unique_layerstack_z,
    order_layerstack,
)
from gdsfactory.simulation.gmsh.uz_xsection_mesh import (
    get_u_bounds_layers,
    get_u_bounds_polygons,
    get_uz_bounds_layers,
    uz_xsection_mesh,
)
from gdsfactory.simulation.gmsh.xy_xsection_mesh import xy_xsection_mesh

__all__ = [
    "mesh_from_polygons",
    "create_physical_mesh",
    "uz_xsection_mesh",
    "xy_xsection_mesh",
    "get_uz_bounds_layers",
    "get_u_bounds_layers",
    "get_u_bounds_polygons",
    "MeshTracker",
    "cleanup_component",
    "fuse_polygons",
    "round_coordinates",
    "to_polygons",
    "tile_shapes",
    "get_layers_at_z",
    "order_layerstack",
    "list_unique_layerstack_z",
    "map_unique_layerstack_z",
    "get_layer_overlaps_z",
]
__version__ = "0.0.2"
