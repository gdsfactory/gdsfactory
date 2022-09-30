from typing import Dict, Optional, Tuple

import numpy as np
import pygmsh

import gdsfactory as gf
from gdsfactory.geometry.boolean import boolean
from gdsfactory.pdk import get_layer_stack
from gdsfactory.tech import LayerStack
from gdsfactory.types import ComponentOrReference, Layer


def get_xsection_bounds_inplane(
    component: ComponentOrReference,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
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
    for layer in component.get_layers():
        bounds_layer = []
        c_layer = component.extract(layer)
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
        bounds_dict[layer] = bounds_layer

    return bounds_dict


def get_xsection_bounds(
    component: ComponentOrReference,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    layer_stack: Optional[LayerStack] = None,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
):
    """Given a component and layer stack, computes the bounding box(es) of each \
            layer in the xsection coordinate system (u,z).

    Args:
        component: Component or ComponentReference.
        xsection_bounds: ( (x1,y1), (x2,y2) ), with x1,y1 beginning point of cross-sectional line and x2,y2 the end
        line_layer: (dummy) layer to put the extraction line on.
        line_width: (dummy) thickness of extraction line. Cannot be 0, should be small (near dbu) for accuracy.

    Returns: Dict containing layer: list pairs, with list a list of bounding box coordinates (u1,u2) in xsection line coordinates
    """
    # Get in-plane cross-sections
    inplane_bounds_dict = get_xsection_bounds_inplane(component, xsection_bounds)

    # Generate full bounding boxes
    layer_stack = layer_stack or get_layer_stack()
    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    exclude_layers = exclude_layers or ()

    outplane_bounds_dict = {}

    for layer, inplane_bounds_list in inplane_bounds_dict.items():
        outplane_bounds_list = []
        for inplane_bounds in inplane_bounds_list:
            if (
                layer not in exclude_layers
                and layer in layer_to_thickness
                and layer in layer_to_zmin
            ):
                height = layer_to_thickness[layer]
                zmin = layer_to_zmin[layer]

                # Get bounding box
                umin = np.min(inplane_bounds)
                umax = np.max(inplane_bounds)
                zmax = zmin + height
                outplane_bounds_list.append(
                    {"umin": umin, "umax": umax, "zmin": zmin, "zmax": zmax}
                )

        outplane_bounds_dict[layer] = outplane_bounds_list

    return outplane_bounds_dict


def mesh2D(
    component: ComponentOrReference,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    base_resolution: float = 0.2,
    refine_resolution: Optional[Dict[Layer, float]] = None,
    padding: Tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0),
    layer_stack: Optional[LayerStack] = None,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
):
    """Returns gmsh 2D geometry of component along cross-sectional line (x1,y1), (x2,y2).

    Args:
        component: Component or ComponentReference.
        xsection_bounds: ( (x1,y1), (x2,y2) ), with x1,y1 beginning point of cross-sectional line and x2,y2 the end.
        base_resolution: background mesh resolution (um).
        refine_resolution: feature mesh resolution (um); layer dependent via a dict (default to base_resolution).
        padding: amount (left, right, bottom, up) to enlarge simulation region beyond features (um).

    """
    layer_stack = layer_stack or get_layer_stack()
    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    exclude_layers = exclude_layers or ()

    geometry = pygmsh.geo.Geometry()

    model = geometry.__enter__()

    # Find extremal coordinates
    bounds_dict = get_xsection_bounds(component, xsection_bounds)
    polygons = [polygon for polygons in bounds_dict.values() for polygon in polygons]
    umin = min(polygon["umin"] for polygon in polygons) - padding[0]
    umax = max(polygon["umax"] for polygon in polygons) + padding[1]
    zmin = min(polygon["zmin"] for polygon in polygons) - padding[2]
    zmax = max(polygon["zmax"] for polygon in polygons) + padding[3]

    # Background oxide
    points = [
        model.add_point([umin, zmin], mesh_size=base_resolution),
        model.add_point([umax, zmin], mesh_size=base_resolution),
        model.add_point([umax, zmax], mesh_size=base_resolution),
        model.add_point([umin, zmax], mesh_size=base_resolution),
    ]
    channel_lines = [
        model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
    ]
    channel_loop = model.add_curve_loop(channel_lines)

    # Add layers
    blocks = []
    for layer in component.get_layers():
        if (
            layer not in exclude_layers
            and layer in layer_to_thickness
            and layer in layer_to_zmin
        ):
            if bounds_dict[layer] == []:
                continue
            blocks_layer = []
            for i, bounds in enumerate(bounds_dict[layer]):
                points = [
                    [bounds["umin"], bounds["zmin"]],
                    [bounds["umin"], bounds["zmax"]],
                    [bounds["umax"], bounds["zmax"]],
                    [bounds["umax"], bounds["zmin"]],
                ]
                polygon = model.add_polygon(
                    points,
                    mesh_size=refine_resolution[layer]
                    if refine_resolution
                    else base_resolution,
                )
                model.add_physical(polygon, f"{layer}_{i}")
                blocks.append(polygon)
                blocks_layer.append(polygon)
            model.add_physical(blocks_layer, f"{layer}")
    plane_surface = model.add_plane_surface(channel_loop, holes=blocks)

    model.add_physical(plane_surface, "oxide")
    model.add_physical([channel_lines[0]], "left")
    model.add_physical([channel_lines[1]], "bottom")
    model.add_physical([channel_lines[2]], "right")
    model.add_physical([channel_lines[3]], "top")

    geometry.generate_mesh(dim=2, verbose=True)

    return geometry


if __name__ == "__main__":
    heaters = gf.Component("heaters")
    heater1 = gf.components.straight_heater_metal(length=50)
    heater2 = gf.components.straight_heater_metal(length=50).move([0, 1])

    heaters = gf.Component()
    heaters << heater1
    heaters << heater2
    heaters.show()

    geometry = mesh2D(
        heaters,
        [(25, -2), (25, 25)],
        exclude_layers=[(1, 10)],
        refine_resolution={(1, 0): 0.02, (47, 0): 0.07},
    )

    import gmsh

    gmsh.write("mesh.msh")
    gmsh.clear()
    geometry.__exit__()

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
