from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

import gdsfactory as gf
from gdsfactory.geometry.boolean import boolean
from gdsfactory.simulation.gmsh.mesh import mesh_from_polygons
from gdsfactory.simulation.gmsh.parse_gds import to_polygons
from gdsfactory.simulation.gmsh.parse_layerstack import order_layerstack
from gdsfactory.tech import LayerStack
from gdsfactory.types import ComponentOrReference, Layer


def get_xsection_bounds_inplane(
    component: ComponentOrReference,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    layer_stack: Optional[LayerStack] = None,
    line_layer: Optional[Layer] = 99,
    line_width: Optional[float] = 0.01,
):
    """Given a component c and two coordinates (x1,y1), (x2,y2), computes the \
        bounding box(es) of each layer in the xsection coordinate system (u).

    Uses gdsfactory boolean and component.extract

    Args:
        component: Component or ComponentReference.
        xsection_bounds: ( (x1,y1), (x2,y2) ), with x1,y1 beginning point of cross-sectional line and x2,y2 the end.
        line_layer: (dummy) layer to put the extraction line on.
        line_width: (dummy) thickness of extraction line. Cannot be 0, should be small (near dbu) for accuracy.

    Returns: Dict containing layer(list pairs, with list a list of bounding box coordinates (u1,u2))
        in xsection line coordinates.
    """
    # Create line component for bool
    P = gf.Path(xsection_bounds)
    X = gf.CrossSection(width=line_width, layer=line_layer)
    line = gf.path.extrude(P, X)

    # Ref line vector
    ref_vector_origin = np.array(xsection_bounds[0])

    # For each layer in component, extract point where line intersects
    # Choose intersecting polygon edge with maximal (including sign) dot product to reference line
    bounds_dict = {}
    layer_dict = layer_stack.to_dict()
    for layername, layer in layer_dict.items():
        bounds_layer = []
        c_layer = component.extract(layer["layer"])
        layer_boolean = boolean(c_layer, line, "and")
        polygons = layer_boolean.get_polygons()
        for polygon in polygons:
            # Min point is midpoint between two closest vertices to origin
            distances_origin = np.linalg.norm(polygon - ref_vector_origin, axis=1)
            distances_sort_index = np.argsort(distances_origin)
            min_point = (
                polygon[distances_sort_index[0]] + polygon[distances_sort_index[1]]
            ) / 2
            max_point = (
                polygon[distances_sort_index[2]] + polygon[distances_sort_index[3]]
            ) / 2

            umin = np.linalg.norm(min_point - ref_vector_origin)
            umax = np.linalg.norm(max_point - ref_vector_origin)

            bounds_layer.append((umin, umax))
        bounds_dict[layername] = bounds_layer

    return bounds_dict


def get_xsection_bound_polygons(
    component: ComponentOrReference,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    layer_stack: Optional[LayerStack] = None,
):
    """Given a component and layer stack, computes the bounding box(es) of each \
            layer in the xsection coordinate system (u,z).

    Args:
        component: Component or ComponentReference.
        xsection_bounds: ( (x1,y1), (x2,y2) ), with x1,y1 beginning point of cross-sectional line and x2,y2 the end
        line_layer: (dummy) layer to put the extraction line on.
        line_width: (dummy) thickness of extraction line. Cannot be 0, should be small (near dbu) for accuracy.

    Returns: Dict containing layer: polygon pairs, with (u1,u2) in xsection line coordinates
    """
    # Get in-plane cross-sections
    inplane_bounds_dict = get_xsection_bounds_inplane(
        component, xsection_bounds, layer_stack
    )

    # Generate full bounding boxes
    layer_dict = layer_stack.to_dict()

    outplane_bounds_dict = {}

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
):

    # Find coordinates
    bounds_dict = get_xsection_bound_polygons(component, xsection_bounds, layerstack)

    # Create polygons from bounds and layers
    layer_order = order_layerstack(layerstack)
    shapes = OrderedDict()
    ordered_layers = set(layer_order).intersection(bounds_dict.keys())
    for layer in ordered_layers:
        layer_shapes = []
        for polygon in bounds_dict[layer]:
            layer_shapes.append(polygon)
        shapes[layer] = MultiPolygon(to_polygons(layer_shapes))

    # Add background polygon
    if background_tag is not None:
        bounds = unary_union([shape for shape in shapes.values()]).bounds
        shapes[background_tag] = Polygon(
            [
                [bounds[0] - background_padding[0], bounds[1] - background_padding[1]],
                [bounds[0] - background_padding[0], bounds[3] + background_padding[3]],
                [bounds[2] + background_padding[2], bounds[3] + background_padding[3]],
                [bounds[2] + background_padding[2], bounds[1] - background_padding[1]],
            ]
        )

    # Mesh
    return mesh_from_polygons(
        shapes,
        resolutions=resolutions,
        filename="mesh.msh",
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
