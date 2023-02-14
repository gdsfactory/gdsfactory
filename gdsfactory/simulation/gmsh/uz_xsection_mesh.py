from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

import gdsfactory as gf
from gdsfactory.simulation.gmsh.mesh import mesh_from_polygons
from gdsfactory.simulation.gmsh.parse_component import (
    create_2D_surface_interface,
    merge_by_material_func,
    process_buffers,
)
from gdsfactory.simulation.gmsh.parse_gds import cleanup_component, to_polygons
from gdsfactory.simulation.gmsh.parse_layerstack import (
    list_unique_layerstack_z,
    order_layerstack,
)
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentOrReference


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
        if intersection := polygon.intersection(line):
            for entry in (
                intersection.geoms if hasattr(intersection, "geoms") else [intersection]
            ):
                bounds = entry.bounds
                p1 = Point([bounds[0], bounds[1]])
                p2 = Point([bounds[2], bounds[3]])
                return_list.append([linestart.distance(p1), linestart.distance(p2)])
    return return_list


def get_u_bounds_layers(
    layer_polygons_dict: Dict[Tuple(str, str, str), MultiPolygon],
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
    for layername, (
        gds_layername,
        next_layername,
        polygons,
        next_polygons,
    ) in layer_polygons_dict.items():
        bounds_dict[layername] = []
        bounds = get_u_bounds_polygons(polygons, xsection_bounds)
        next_bounds = get_u_bounds_polygons(next_polygons, xsection_bounds)
        if bounds:
            bounds_dict[layername] = (
                gds_layername,
                next_layername,
                bounds,
                next_bounds,
            )

    return bounds_dict


def get_uz_bounds_layers(
    layer_polygons_dict: Dict[str, Tuple[str, MultiPolygon, MultiPolygon]],
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

    # Remove empty entries
    inplane_bounds_dict = {
        key: value for (key, value) in inplane_bounds_dict.items() if value
    }

    for layername, (
        gds_layername,
        next_layername,
        inplane_bounds_list,
        next_inplane_bounds_list,
    ) in inplane_bounds_dict.items():
        outplane_polygons_list = []
        for inplane_bounds, next_inplane_bounds in zip(
            inplane_bounds_list, next_inplane_bounds_list
        ):
            zmin = layer_dict[layername]["zmin"]
            zmax = layer_dict[next_layername]["zmin"]

            # Get bounding box
            umin_zmin = np.min(inplane_bounds)
            umax_zmin = np.max(inplane_bounds)
            umin_zmax = np.min(next_inplane_bounds)
            umax_zmax = np.max(next_inplane_bounds)

            points = [
                [umin_zmin, zmin],
                [umin_zmax, zmax],
                [umax_zmax, zmax],
                [umax_zmin, zmin],
            ]
            outplane_polygons_list.append(Polygon(points))

        outplane_bounds_dict[layername] = (gds_layername, outplane_polygons_list)

    return outplane_bounds_dict


def uz_xsection_mesh(
    component: ComponentOrReference,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    layerstack: LayerStack,
    resolutions: Optional[Dict] = None,
    mesh_scaling_factor: float = 1.0,
    default_resolution_min: float = 0.01,
    default_resolution_max: float = 0.5,
    background_tag: Optional[str] = None,
    background_padding: Tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0),
    filename: Optional[str] = None,
    global_meshsize_array: Optional[np.array] = None,
    global_meshsize_interpolant_func: Optional[callable] = NearestNDInterpolator,
    extra_shapes_dict: Optional[OrderedDict] = None,
    merge_by_material: Optional[bool] = False,
    interface_surfaces: Optional[Dict[str, Tuple(float, float)]] = None,
    round_tol: int = 3,
    simplify_tol: float = 1e-2,
    **kwargs,
):
    """Mesh uz cross-section of component along line u = [[x1,y1] , [x2,y2]].

    Args:
        component (Component): gdsfactory component to mesh
        xsection_bounds (Tuple): Tuple [[x1,y1] , [x2,y2]] parametrizing the line u
        layerstack (LayerStack): gdsfactory LayerStack to parse
        resolutions (Dict): Pairs {"layername": {"resolution": float, "distance": "float}} to roughly control mesh refinement
        mesh_scaling_factor (float): factor multiply mesh geometry by
        default_resolution_min (float): gmsh minimal edge length
        default_resolution_max (float): gmsh maximal edge length
        background_tag (str): name of the background layer to add (default: no background added)
        background_padding (Tuple): [xleft, ydown, xright, yup] distances to add to the components and to fill with background_tag
        filename (str, path): where to save the .msh file
        global_meshsize_array: np array [x,y,z,lc] to parametrize the mesh
        global_meshsize_interpolant_func: interpolating function for global_meshsize_array
        extra_shapes_dict: Optional[OrderedDict] = OrderedDict of {key: geo} with key a label and geo a shapely (Multi)Polygon or (Multi)LineString of extra shapes to override component
        merge_by_material: boolean, if True will merge polygons from layers with the same layer.material. Physical keys will be material in this case.
        round_tol: during gds --> mesh conversion cleanup, number of decimal points at which to round the gdsfactory/shapely points before introducing to gmsh
        simplify_tol: during gds --> mesh conversion cleanup, shapely "simplify" tolerance (make it so all points are at least separated by this amount)
    """
    interface_surfaces = interface_surfaces or {}

    # Fuse and cleanup polygons of same layer in case user overlapped them
    layer_polygons_dict = cleanup_component(
        component, layerstack, round_tol, simplify_tol
    )

    # GDS polygons to simulation polygons
    buffered_layer_polygons_dict, buffered_layerstack = process_buffers(
        layer_polygons_dict, layerstack
    )

    # simulation polygons to u-z coordinates along cross-sectional line
    bounds_dict = get_uz_bounds_layers(
        buffered_layer_polygons_dict, xsection_bounds, buffered_layerstack
    )

    # u-z coordinates to gmsh-friendly polygons
    # Remove terminal layers and merge polygons
    layer_order = order_layerstack(layerstack)  # gds layers
    shapes = OrderedDict() if extra_shapes_dict is None else extra_shapes_dict
    for layername in layer_order:
        current_shapes = []
        for _, (gds_name, bounds) in bounds_dict.items():
            if gds_name == layername:
                layer_shapes = list(bounds)
                current_shapes.append(MultiPolygon(to_polygons(layer_shapes)))
        shapes[layername] = unary_union(MultiPolygon(to_polygons(current_shapes)))

    # Add background polygon
    if background_tag is not None:
        # shapes[background_tag] = bounds.buffer(background_padding[0])
        # bounds = unary_union(list(shapes.values())).bounds
        zs = list_unique_layerstack_z(buffered_layerstack)
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

    # Merge by material
    if merge_by_material:
        shapes = merge_by_material_func(shapes, layerstack)

    # Create interface surfaces
    reordered_shapes = OrderedDict()
    for label in shapes.keys():
        if interface_surfaces and label in interface_surfaces.keys():
            buffer_in, buffer_out, simplification = interface_surfaces[label]
            reordered_shapes[f"{label}Int"] = create_2D_surface_interface(
                shapes[label], buffer_in, buffer_out, simplification
            )
        reordered_shapes[label] = shapes[label]

    # Mesh
    return mesh_from_polygons(
        reordered_shapes,
        resolutions=resolutions,
        mesh_scaling_factor=mesh_scaling_factor,
        filename=filename,
        default_resolution_min=default_resolution_min,
        default_resolution_max=default_resolution_max,
        global_meshsize_array=global_meshsize_array,
        global_meshsize_interpolant_func=global_meshsize_interpolant_func,
    )


if __name__ == "__main__":
    from gdsfactory.pdk import get_layer_stack

    c = gf.component.Component()

    waveguide = c << gf.get_component(gf.components.straight_pin(length=10, taper=None))
    undercut = c << gf.get_component(
        gf.components.rectangle(
            size=(5.0, 5.0),
            layer="UNDERCUT",
            centered=True,
        )
    ).move(destination=[4, 0])
    c.show()

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                "slab90",
                "core",
                "via_contact",
                # "undercut",
                "box",
                # "substrate",
                "clad",
                "metal1",
            )  # "slab90", "via_contact")#"via_contact") # "slab90", "core"
        }
    )

    resolutions = {}
    resolutions["si"] = {"resolution": 0.02, "distance": 0.1}
    resolutions["siInt"] = {"resolution": 0.0003, "distance": 0.1}
    # resolutions["sio2"] = {"resolution": 0.5, "distance": 1}
    resolutions["Aluminum"] = {"resolution": 0.1, "distance": 1}

    # resolutions = {}
    # resolutions["core"] = {"resolution": 0.02, "distance": 0.1}
    # resolutions["coreInt"] = {"resolution": 0.0001, "distance": 0.1}
    # resolutions["slab90"] = {"resolution": 0.05, "distance": 0.1}
    # # resolutions["sio2"] = {"resolution": 0.5, "distance": 1}
    # resolutions["via_contact"] = {"resolution": 0.1, "distance": 1}

    geometry = uz_xsection_mesh(
        c,
        [(4, -15), (4, 15)],
        filtered_layerstack,
        resolutions=resolutions,
        # background_tag="Oxide",
        filename="mesh.msh",
        merge_by_material=True,
        interface_surfaces={"si": (0.002, 0.002, 0.0002)},
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
