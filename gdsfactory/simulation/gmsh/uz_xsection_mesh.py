from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

import gdsfactory as gf
from gdsfactory.simulation.gmsh.mesh import mesh_from_polygons
from gdsfactory.simulation.gmsh.parse_gds import cleanup_component, to_polygons
from gdsfactory.simulation.gmsh.parse_layerstack import (
    list_unique_layerstack_z,
    order_layerstack,
)
from gdsfactory.tech import LayerStack
from gdsfactory.types import ComponentOrReference


def get_u_bounds_polygons(
    polygons: Union[MultiPolygon, List[Polygon]],
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
):
    """Performs the bound extraction given a (Multi)Polygon or [Polygon] and cross-sectional line coordinates.

    Args:
        layer_polygons_dict: dict containing layernames: shapely polygons pairs
        xsection_bounds: ( (x1,y1), (x2,y2) ), with x1,y1 beginning point of cross-sectional line and x2,y2 the end.

    Returns: list of bounding box coordinates (u1,u2)) in xsection line coordinates (distance from xsection_bounds[0]).
    """
    line = LineString(xsection_bounds)
    linestart = Point(xsection_bounds[0])

    return_list = []
    for polygon in polygons.geoms if hasattr(polygons, "geoms") else [polygons]:
        intersection = polygon.intersection(line).bounds
        if intersection:
            p1 = Point([intersection[0], intersection[1]])
            p2 = Point([intersection[2], intersection[3]])
            return_list.append([linestart.distance(p1), linestart.distance(p2)])
    return return_list


def get_u_bounds_layers(
    layer_polygons_dict: Dict[str, MultiPolygon],
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
):
    """Given a layer_polygons_dict and two coordinates (x1,y1), (x2,y2), computes the \
        bounding box(es) of each layer in the xsection coordinate system (u).

    Args:
        layer_polygons_dict: dict containing layernames: shapely polygons pairs
        xsection_bounds: ( (x1,y1), (x2,y2) ), with x1,y1 beginning point of cross-sectional line and x2,y2 the end.

    Returns: Dict containing layer(list pairs, with list a list of bounding box coordinates (u1,u2))
        in xsection line coordinates.
    """
    bounds_dict = {}
    for layername, polygons in layer_polygons_dict.items():
        bounds_dict[layername] = []
        bounds = get_u_bounds_polygons(polygons, xsection_bounds)
        if bounds:
            bounds_dict[layername] = bounds

    return bounds_dict


def get_uz_bounds_layers(
    layer_polygons_dict: Dict[str, MultiPolygon],
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    layerstack: LayerStack,
):
    """Given a component and layer stack, computes the bounding box(es) of each \
            layer in the xsection coordinate system (u,z).

    Args:
        component: Component or ComponentReference.
        xsection_bounds: ( (x1,y1), (x2,y2) ), with x1,y1 beginning point of cross-sectional line and x2,y2 the end

    Returns: Dict containing layer: polygon pairs, with (u1,u2) in xsection line coordinates
    """
    # Get in-plane cross-sections
    inplane_bounds_dict = get_u_bounds_layers(layer_polygons_dict, xsection_bounds)

    outplane_bounds_dict = {}

    layer_dict = layerstack.to_dict()
    for layername, inplane_bounds_list in inplane_bounds_dict.items():
        outplane_polygons_list = []
        for inplane_bounds in inplane_bounds_list:
            height = layer_dict[layername]["thickness"]
            zmin = layer_dict[layername]["zmin"]
            sidewall_angle = layer_dict[layername]["sidewall_angle"]

            # Get bounding box
            umin = np.min(inplane_bounds)
            umax = np.max(inplane_bounds)
            zmax = zmin + height

            points = [
                [umin, zmin],
                [umin + height * (np.tan(np.radians(sidewall_angle))), zmax],
                [umax - height * (np.tan(np.radians(sidewall_angle))), zmax],
                [umax, zmin],
            ]
            outplane_polygons_list.append(Polygon(points))

        outplane_bounds_dict[layername] = outplane_polygons_list

    return outplane_bounds_dict


def uz_xsection_mesh(
    component: ComponentOrReference,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    layerstack: LayerStack,
    resolutions: Optional[Dict],
    default_resolution_min: float = 0.01,
    default_resolution_max: float = 0.5,
    background_tag: Optional[str] = None,
    background_padding: Tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0),
    filename: Optional[str] = None,
):
    # Fuse and cleanup polygons of same layer in case user overlapped them
    layer_polygons_dict = cleanup_component(component, layerstack)

    # Find coordinates
    bounds_dict = get_uz_bounds_layers(layer_polygons_dict, xsection_bounds, layerstack)

    # Create polygons from bounds and layers
    layer_order = order_layerstack(layerstack)
    shapes = OrderedDict()
    for layer in layer_order:
        layer_shapes = list(bounds_dict[layer])
        shapes[layer] = MultiPolygon(to_polygons(layer_shapes))

    # Add background polygon
    # TODO: buffer the union instead of adding a square
    if background_tag is not None:
        # shapes[background_tag] = bounds.buffer(background_padding[0])
        # bounds = unary_union(list(shapes.values())).bounds
        zs = list_unique_layerstack_z(layerstack)
        zmin = np.min(zs)
        zmax = np.max(zs)
        shapes[background_tag] = Polygon(
            [
                [-1 * background_padding[0], zmin - background_padding[1]],
                [-1 * background_padding[0], zmax + background_padding[3]],
                [
                    np.linalg.norm(
                        np.array(xsection_bounds[1]) - np.array(xsection_bounds[0])
                    )
                    + background_padding[2],
                    zmax + background_padding[3],
                ],
                [
                    np.linalg.norm(
                        np.array(xsection_bounds[1]) - np.array(xsection_bounds[0])
                    )
                    + background_padding[2],
                    zmin - background_padding[1],
                ],
            ]
        )

    # Mesh
    return mesh_from_polygons(
        shapes,
        resolutions=resolutions,
        filename=filename,
        default_resolution_min=default_resolution_min,
        default_resolution_max=default_resolution_max,
    )


if __name__ == "__main__":

    from gdsfactory.tech import get_layer_stack_generic

    waveguide = gf.components.straight_pin(length=10, taper=None)
    waveguide.show()

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
                "via_contact",
                # "metal2",
            )  # "slab90", "via_contact")#"via_contact") # "slab90", "core"
        }
    )

    resolutions = {}
    resolutions["core"] = {"resolution": 0.05, "distance": 2}
    resolutions["slab90"] = {"resolution": 0.03, "distance": 1}
    resolutions["via_contact"] = {"resolution": 0.1, "distance": 1}

    geometry = uz_xsection_mesh(
        waveguide,
        [(4, -15), (4, 15)],
        filtered_layerstack,
        resolutions=resolutions,
        background_tag="Oxide",
        filename="mesh.msh",
    )

    import meshio

    mesh_from_file = meshio.read("mesh.msh")

    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:, :2] if prune_z else mesh.points
        return meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data]},
        )

    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write("facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("mesh.xdmf", triangle_mesh)
