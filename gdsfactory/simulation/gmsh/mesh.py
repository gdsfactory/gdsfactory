from __future__ import annotations

from collections import OrderedDict
from itertools import combinations
from typing import Dict, Optional

import gmsh
import pygmsh
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from gdsfactory.simulation.gmsh.meshtracker import MeshTracker
from gdsfactory.simulation.gmsh.parse_gds import break_line, tile_shapes


def mesh_from_polygons(
    shapes_dict: OrderedDict,
    resolutions: Optional[Dict[str, float]] = None,
    default_resolution_min: float = 0.01,
    default_resolution_max: float = 0.5,
    filename: Optional[str] = None,
):
    """Return a 2D mesh from an ordered dict of shapely polygons."""
    with pygmsh.occ.geometry.Geometry() as geometry:

        gmsh.initialize()

        geometry.characteristic_length_min = default_resolution_min
        geometry.characteristic_length_max = default_resolution_max

        model = geometry.__enter__()

        # Break up shapes in order so that plane is tiled with non-overlapping layers
        shapes_tiled_dict = tile_shapes(shapes_dict)

        # Break up lines and polygon edges so that plane is tiled with no partially overlapping line segments
        polygons_broken_dict = OrderedDict()
        lines_broken_dict = OrderedDict()
        for _first_index, (first_name, _init_first_shapes) in enumerate(
            shapes_dict.items()
        ):
            first_shapes = shapes_tiled_dict[first_name]
            broken_shapes = []
            for first_shape in (
                first_shapes.geoms if hasattr(first_shapes, "geoms") else [first_shapes]
            ):
                # First line exterior
                first_exterior_line = (
                    LineString(first_shape.exterior)
                    if first_shape.type == "Polygon"
                    else first_shape
                )
                for _second_index, (second_name, second_shapes) in enumerate(
                    shapes_dict.items()
                ):
                    # Do not compare to itself
                    if second_name == first_name:
                        continue
                    else:
                        second_shapes = shapes_tiled_dict[second_name]
                        for second_shape in (
                            second_shapes.geoms
                            if hasattr(second_shapes, "geoms")
                            else [second_shapes]
                        ):
                            # Second line exterior
                            second_exterior_line = (
                                LineString(second_shape.exterior)
                                if second_shape.type == "Polygon"
                                else second_shape
                            )
                            first_exterior_line = break_line(
                                first_exterior_line, second_exterior_line
                            )
                            # Second line interiors
                            for second_interior_line in (
                                second_shape.interiors
                                if second_shape.type == "Polygon"
                                else []
                            ):
                                second_interior_line = LineString(second_interior_line)
                                first_exterior_line = break_line(
                                    first_exterior_line, second_interior_line
                                )
                # First line interiors
                if first_shape.type == "Polygon" or first_shape.type == "MultiPolygon":
                    first_shape_interiors = []
                    for first_interior_line in first_shape.interiors:
                        first_interior_line = LineString(first_interior_line)
                        for _second_index, (second_name, second_shapes) in enumerate(
                            shapes_dict.items()
                        ):
                            if second_name == first_name:
                                continue
                            else:
                                second_shapes = shapes_tiled_dict[second_name]
                                for second_shape in (
                                    second_shapes.geoms
                                    if hasattr(second_shapes, "geoms")
                                    else [second_shapes]
                                ):
                                    # Exterior
                                    second_exterior_line = (
                                        LineString(second_shape.exterior)
                                        if second_shape.type == "Polygon"
                                        else second_shape
                                    )
                                    first_interior_line = break_line(
                                        first_interior_line, second_exterior_line
                                    )
                                    # Interiors
                                    for second_interior_line in (
                                        second_shape.interiors
                                        if second_shape.type == "Polygon"
                                        else []
                                    ):
                                        second_interior_line = LineString(
                                            second_interior_line
                                        )
                                        first_interior_line = break_line(
                                            first_interior_line, second_interior_line
                                        )
                        first_shape_interiors.append(first_interior_line)
                if first_shape.type == "Polygon" or first_shape.type == "MultiPolygon":
                    broken_shapes.append(
                        Polygon(first_exterior_line, holes=first_shape_interiors)
                    )
                else:
                    broken_shapes.append(LineString(first_exterior_line))
            if broken_shapes:
                if first_shape.type == "Polygon" or first_shape.type == "MultiPolygon":
                    polygons_broken_dict[first_name] = (
                        MultiPolygon(broken_shapes)
                        if len(broken_shapes) > 1
                        else broken_shapes[0]
                    )
                else:
                    lines_broken_dict[first_name] = (
                        MultiLineString(broken_shapes)
                        if len(broken_shapes) > 1
                        else broken_shapes[0]
                    )

        # Add lines, reusing line segments
        meshtracker = MeshTracker(model=model, atol=1e-3)
        for line_name, line in lines_broken_dict.items():
            meshtracker.add_get_xy_line(line, line_name)

        # Add surfaces, reusing lines to simplify at early stage
        for polygon_name, polygons in polygons_broken_dict.items():
            gmsh_surfaces = []
            for polygon in polygons if hasattr(polygons, "geoms") else [polygons]:
                gmsh_surface = meshtracker.add_xy_surface(polygon, f"{polygon_name}")
                gmsh_surfaces.append(gmsh_surface)
            meshtracker.model.add_physical(gmsh_surfaces, f"{polygon_name}")

        # Refinement in surfaces
        n = 0
        refinement_fields = []
        for label, resolution in resolutions.items():
            # Inside surface
            mesh_resolution = resolution["resolution"]
            gmsh.model.mesh.field.add("MathEval", n)
            gmsh.model.mesh.field.setString(n, "F", f"{mesh_resolution}")
            gmsh.model.mesh.field.add("Restrict", n + 1)
            gmsh.model.mesh.field.setNumber(n + 1, "InField", n)
            gmsh.model.mesh.field.setNumbers(
                n + 1,
                "SurfacesList",
                meshtracker.get_gmsh_xy_surfaces_from_label(label),
            )
            # Around surface
            mesh_distance = resolution["distance"]
            gmsh.model.mesh.field.add("Distance", n + 2)
            gmsh.model.mesh.field.setNumbers(
                n + 2, "CurvesList", meshtracker.get_gmsh_xy_lines_from_label(label)
            )
            gmsh.model.mesh.field.setNumber(n + 2, "Sampling", 100)
            gmsh.model.mesh.field.add("Threshold", n + 3)
            gmsh.model.mesh.field.setNumber(n + 3, "InField", n + 2)
            gmsh.model.mesh.field.setNumber(n + 3, "SizeMin", mesh_resolution)
            gmsh.model.mesh.field.setNumber(n + 3, "SizeMax", default_resolution_max)
            gmsh.model.mesh.field.setNumber(n + 3, "DistMin", 0)
            gmsh.model.mesh.field.setNumber(n + 3, "DistMax", mesh_distance)
            # Save and increment
            refinement_fields.append(n + 1)
            refinement_fields.append(n + 3)
            n += 4

        # Use the smallest element size overall
        gmsh.model.mesh.field.add("Min", n)
        gmsh.model.mesh.field.setNumbers(n, "FieldsList", refinement_fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(n)

        gmsh.model.mesh.MeshSizeFromPoints = 0
        gmsh.model.mesh.MeshSizeFromCurvature = 0
        gmsh.model.mesh.MeshSizeExtendFromBoundary = 0

        # Tag all interfacial lines
        for surface1, surface2 in combinations(polygons_broken_dict.keys(), 2):
            interfaces = []
            for index, line in enumerate(meshtracker.gmsh_xy_segments):
                if (
                    meshtracker.xy_segments_main_labels[index] == surface1
                    and meshtracker.xy_segments_secondary_labels[index] == surface2
                ) or (
                    meshtracker.xy_segments_main_labels[index] == surface2
                    and meshtracker.xy_segments_secondary_labels[index] == surface1
                ):
                    interfaces.append(line)
            if interfaces:
                model.add_physical(interfaces, f"{surface1}___{surface2}")

        mesh = geometry.generate_mesh(dim=2, verbose=True)

        if filename:
            gmsh.write(f"{filename}")

        return mesh


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
    resolutions["core_clad"] = {"resolution": 0.01, "distance": 0.5}
    resolutions["clad_box"] = {"resolution": 0.01, "distance": 0.5}
    resolutions["bottom_edge"] = {"resolution": 0.05, "distance": 0.5}
    resolutions["left_edge"] = {"resolution": 0.05, "distance": 0.5}
    # resolutions["clad"] = {"resolution": 0.1, "dist_min": 0.01, "dist_max": 0.3}

    mesh = mesh_from_polygons(shapes, resolutions, filename="mesh.msh")

    # gmsh.write("mesh.msh")
    # gmsh.clear()
    # mesh.__exit__()

    import meshio

    mesh_from_file = meshio.read("mesh.msh")

    def create_mesh(mesh, cell_type, prune_z=True):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points
        return meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data]},
        )

    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write("facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("mesh.xdmf", triangle_mesh)
