from __future__ import annotations

from collections import OrderedDict
from itertools import combinations, product
from typing import Dict, Optional

import gmsh
import meshio
import numpy as np
import pygmsh
from scipy.interpolate import NearestNDInterpolator
from shapely.geometry import LineString, Point, Polygon

from gdsfactory.simulation.gmsh.break_geometry import break_geometry
from gdsfactory.simulation.gmsh.meshtracker import MeshTracker
from gdsfactory.simulation.gmsh.refine import (
    global_callback_refinement,
    surface_interface_refinement,
)


def define_entities(model, shapes_dict: OrderedDict, atol=1e-3):
    """Adds the polygons and lines from a "shapes_dict" as physical entities in the pygmsh model "model".

    Args:
        model: pygmsh model to define the entities into.
        shapes_dict: OrderedDict containing shapes to mesh as values, and keys as physical tags.

    Returns:
        model: updated pygmsh model.
        meshtracker: meshtracker object containing the mapping between labels, shapely objects, and gmsh objects.
    """
    # Break up lines and polygon edges so that plane is tiled with no partially overlapping line segments
    polygons_broken_dict, lines_broken_dict = break_geometry(shapes_dict)
    # Instantiate shapely to gmsh map
    meshtracker = MeshTracker(model=model, atol=atol)
    # Add lines, reusing line segments
    model, meshtracker = add_lines(model, meshtracker, lines_broken_dict)
    # Add surfaces, reusing lines
    model, meshtracker = add_surfaces(model, meshtracker, polygons_broken_dict)
    # Add interfaces
    model, meshtracker = tag_interfaces(model, meshtracker, polygons_broken_dict)
    # Synchronize
    model.synchronize()

    return model, meshtracker


def add_lines(model, meshtracker, lines_broken_dict):
    """Add lines, reusing segments via meshtracker.

    Args:
        model: pygmsh model
        meshtracker: meshtracker object
        lines_broken_dict: dict

    Returns:
        model: updated pygmsh model
        meshtracker: updated meshtracker object
    """
    for line_name, line in lines_broken_dict.items():
        meshtracker.add_get_xy_line(line, line_name)

    return model, meshtracker


def add_surfaces(model, meshtracker, polygons_broken_dict):
    """Add surfaces of polygons_broken_dict to model, reusing lines via meshtracker.

    Args:
        model: pygmsh model
        meshtracker: meshtracker object
        polygons_broken_dict: dict of

    Returns:
        model: updated pygmsh model
        meshtracker: updated meshtracker object
    """
    for polygon_name, polygons in polygons_broken_dict.items():
        gmsh_surfaces = []
        for polygon in polygons.geoms if hasattr(polygons, "geoms") else [polygons]:
            gmsh_surface = meshtracker.add_xy_surface(polygon, f"{polygon_name}")
            gmsh_surfaces.append(gmsh_surface)
        meshtracker.model.add_physical(gmsh_surfaces, f"{polygon_name}")

    return model, meshtracker


def tag_interfaces(model, meshtracker, polygons_broken_dict):
    """Tag all interfacial lines between polygons as logged in meshtracker.

    Args:
        model: pygmsh model
        meshtracker: meshtracker object
        polygons_broken_dict: dict of

    Returns:
        model: updated pygmsh model
        meshtracker: updated meshtracker object
    """
    for surface1, surface2 in combinations(polygons_broken_dict.keys(), 2):
        if interfaces := [
            line
            for index, line in enumerate(meshtracker.gmsh_xy_segments)
            if (
                meshtracker.xy_segments_main_labels[index] == surface1
                and meshtracker.xy_segments_secondary_labels[index] == surface2
            )
            or (
                meshtracker.xy_segments_main_labels[index] == surface2
                and meshtracker.xy_segments_secondary_labels[index] == surface1
            )
        ]:
            model.add_physical(interfaces, f"{surface1}___{surface2}")

    return model, meshtracker


def mesh_from_polygons(
    shapes_dict: OrderedDict,
    resolutions: Optional[Dict[str, float]] = None,
    mesh_scaling_factor: float = 1.0,
    default_resolution_min: float = 0.01,
    default_resolution_max: float = 0.5,
    filename: Optional[str] = None,
    global_meshsize_array: Optional[np.array] = None,
    global_meshsize_interpolant_func: Optional[callable] = NearestNDInterpolator,
    verbosity: Optional[bool] = False,
    atol: Optional[float] = 1e-4,
):
    """Return a 2D mesh from an ordered dict of shapely polygons.

    Args:
        shapes_dict: OrderedDict containing shapes to mesh.
        resolutions: Dict with same keys as shapes_dict. containing another dict with resolution settings:
            resolution: density of the mesh
            distance: "dropoff" distance of this resolution away from the surface
        default_resolution_min: gmsh default smallest characteristic length
        default_resolution_max: gmsh default largest characteristic length
        filename: to save the mesh
        global_meshsize_array: array [x,y,z,lc] defining local mesh sizes. Not used if None
        global_meshsize_interpolant: interpolation function for array [x,y,z,lc]. Default scipy.interpolate.NearestNDInterpolator
        verbosity: boolean, gmsh stdout as it meshes
        atol: tolerance used to establish equivalency between vertices
    """
    global_meshsize_callback_bool = global_meshsize_array is not None

    geometry = pygmsh.occ.geometry.Geometry()
    model = geometry.__enter__()

    geometry.characteristic_length_min = default_resolution_min
    geometry.characteristic_length_max = default_resolution_max

    # Define geometry
    (
        model,
        meshtracker,
    ) = define_entities(model, shapes_dict, atol)

    # Synchronize
    model.synchronize()

    # Refinement
    if not global_meshsize_callback_bool and resolutions:
        surface_interface_refinement(
            model,
            meshtracker,
            resolutions,
            default_resolution_min,
            default_resolution_max,
        )

    elif global_meshsize_callback_bool:
        global_callback_refinement(
            model,
            global_meshsize_array,
            global_meshsize_interpolant_func,
        )

    # HACK: force shared nodes across interfaces
    gmsh.model.occ.remove_all_duplicates()

    # Perform meshing
    gmsh.option.setNumber("Mesh.ScalingFactor", mesh_scaling_factor)
    mesh = geometry.generate_mesh(dim=2, verbose=verbosity)

    if filename:
        gmsh.write(f"{filename}")

    return mesh


def create_physical_mesh(mesh, cell_type):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points
    return meshio.Mesh(
        points=points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]},
    )


if __name__ == "__main__":
    wmode = 1
    wsim = 2
    hclad = 2
    hbox = 2
    wcore = 0.5
    hcore = 0.22
    offset_core = -0.1
    offset_core2 = 1

    # Lines can be added, which is useful to define boundary conditions at various simulation edges
    left_edge = LineString(
        [Point(-wsim / 2, -hcore / 2 - hbox), Point(-wsim / 2, -hcore / 2 + hclad)]
    )
    right_edge = LineString(
        [Point(wsim / 2, -hcore / 2 - hbox), Point(wsim / 2, -hcore / 2 + hclad)]
    )
    top_edge = LineString(
        [Point(-wsim / 2, -hcore / 2 + hclad), Point(wsim / 2, -hcore / 2 + hclad)]
    )
    bottom_edge = LineString(
        [Point(-wsim / 2, -hcore / 2 - hbox), Point(wsim / 2, -hcore / 2 - hbox)]
    )

    # Polygons not only have an edge, but an interior
    core = Polygon(
        [
            Point(-wcore / 2, -hcore / 2 + offset_core),
            Point(-wcore / 2, hcore / 2 + offset_core),
            Point(wcore / 2, hcore / 2 + offset_core),
            Point(wcore / 2, -hcore / 2 + offset_core),
        ]
    )
    core2 = Polygon(
        [
            Point(-wcore / 2, -hcore / 2 + offset_core2),
            Point(-wcore / 2, hcore / 2 + offset_core2),
            Point(wcore / 2, hcore / 2 + offset_core2),
            Point(wcore / 2, -hcore / 2 + offset_core2),
        ]
    )
    clad = Polygon(
        [
            Point(-wsim / 2, -hcore / 2),
            Point(-wsim / 2, -hcore / 2 + hclad),
            Point(wsim / 2, -hcore / 2 + hclad),
            Point(wsim / 2, -hcore / 2),
        ]
    )
    box = Polygon(
        [
            Point(-wsim / 2, -hcore / 2),
            Point(-wsim / 2, -hcore / 2 - hbox),
            Point(wsim / 2, -hcore / 2 - hbox),
            Point(wsim / 2, -hcore / 2),
        ]
    )

    # The order in which objects are inserted into the OrderedDict determines overrides
    shapes = OrderedDict()
    shapes["left_edge"] = left_edge
    shapes["right_edge"] = right_edge
    shapes["top_edge"] = top_edge
    shapes["bottom_edge"] = bottom_edge
    shapes["core"] = core
    shapes["core2"] = core2
    shapes["clad"] = clad
    shapes["box"] = box

    # The resolution dict is not ordered, and can be used to set mesh resolution at various element
    # The edge of a polygon and another polygon (or entire simulation domain) will form a line object that can be refined independently
    resolutions = {}
    resolutions["core"] = {"resolution": 0.05, "distance": 0}
    resolutions["core___clad"] = {"resolution": 0.01, "distance": 0.5}
    resolutions["clad___box"] = {"resolution": 0.01, "distance": 0.5}
    resolutions["bottom_edge"] = {"resolution": 0.05, "distance": 0.5}
    resolutions["left_edge"] = {"resolution": 0.05, "distance": 0.5}
    # resolutions["clad"] = {"resolution": 0.1, "dist_min": 0.01, "dist_max": 0.3}

    xs = np.linspace(-wsim / 2, wsim / 2, 500)
    ys = np.linspace(-hcore / 2 - hbox, -hcore / 2 + hclad, 500)
    global_meshsize_array = []

    ls0 = 0.02
    ls1 = 0.05
    ls2 = 0.005

    r = 0.5
    for x, y in product(xs, ys):
        if np.abs(x) ** 2 + np.abs(y) ** 2 <= r**2:
            global_meshsize_array.append([x, y, 0, ls2])
        elif y > 0:
            global_meshsize_array.append([x, y, 0, ls1])
        else:
            global_meshsize_array.append([x, y, 0, ls0])

    global_meshsize_array = np.array(global_meshsize_array)

    mesh = mesh_from_polygons(
        shapes,
        resolutions,
        mesh_scaling_factor=0.1,
        filename="mesh.msh",
        global_meshsize_array=global_meshsize_array,
        verbosity=True,
    )

    # gmsh.write("mesh.msh")
    # gmsh.clear()
    # mesh.__exit__()

    mesh_from_file = meshio.read("mesh.msh")

    def create_mesh(mesh, cell_type):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points
        return meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data]},
        )

    line_mesh = create_mesh(mesh_from_file, "line")
    meshio.write("facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle")
    meshio.write("mesh.xdmf", triangle_mesh)
