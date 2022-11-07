from collections import OrderedDict
from typing import Dict, Optional

import pygmsh
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, split


class MeshTracker:
    def __init__(self, model, atol=1e-3):
        """Map between shapely and gmsh.

        Shapely is useful for built-in geometry equivalencies and extracting orientation, instead of doing it manually
        We can also track information about the entities using labels (useful for selective mesh refinement later)
        """
        self.shapely_points = []
        self.gmsh_points = []
        self.points_labels = []
        self.shapely_xy_segments = []
        self.gmsh_xy_segments = []
        self.xy_segments_main_labels = []
        self.xy_segments_secondary_labels = []
        self.gmsh_xy_surfaces = []
        self.xy_surfaces_labels = []
        self.model = model
        self.atol = atol

    """Retrieve existing geometry"""

    def get_point_index(self, xy_point):
        for index, shapely_point in enumerate(self.shapely_points):
            if xy_point.equals_exact(shapely_point, self.atol):
                return index
        return None

    def get_xy_segment_index_and_orientation(self, xy_point1, xy_point2):
        xy_line = shapely.geometry.LineString([xy_point1, xy_point2])
        for index, shapely_line in enumerate(self.shapely_xy_segments):
            if xy_line.equals(shapely_line):
                first_xy_line, last_xy_line = xy_line.boundary.geoms
                first_xy, last_xy = shapely_line.boundary.geoms
                if first_xy_line.equals(first_xy):
                    return index, True
                else:
                    return index, False
        return None, 1

    def get_gmsh_points_from_label(self, label):
        indices = [
            idx for idx, value in enumerate(self.points_labels) if value == label
        ]
        entities = []
        for index in indices:
            entities.append(self.gmsh_points[index]._id)
        return entities

    def get_gmsh_xy_lines_from_label(self, label):
        indices = [
            idx
            for idx, value in enumerate(self.xy_segments_main_labels)
            if value == label
        ]
        entities = []
        for index in indices:
            entities.append(self.gmsh_xy_segments[index]._id)
        return entities

    def get_gmsh_xy_surfaces_from_label(self, label):
        indices = [
            idx for idx, value in enumerate(self.xy_surfaces_labels) if value == label
        ]
        entities = []
        for index in indices:
            entities.append(self.gmsh_xy_surfaces[index]._id)
        return entities

    """Channel loop utilities (no need to track)"""

    def xy_channel_loop_from_vertices(self, vertices, label):
        edges = []
        for vertex1, vertex2 in [
            (vertices[i], vertices[i + 1]) for i in range(0, len(vertices) - 1)
        ]:
            gmsh_line, orientation = self.add_get_xy_segment(vertex1, vertex2, label)
            if orientation:
                edges.append(gmsh_line)
            else:
                edges.append(-gmsh_line)
        channel_loop = self.model.add_curve_loop(edges)
        return channel_loop

    """Adding geometry"""

    def add_get_point(self, shapely_xy_point, label=None):
        """Add a shapely point to the gmsh model, or retrieve the existing gmsh model points with equivalent coordinates (within tol.).

        Args:
            shapely_xy_point (shapely.geometry.Point): x, y coordinates
            resolution (float): gmsh resolution at that point
        """
        index = self.get_point_index(shapely_xy_point)
        if index is not None:
            gmsh_point = self.gmsh_points[index]
        else:
            gmsh_point = self.model.add_point([shapely_xy_point.x, shapely_xy_point.y])
            self.shapely_points.append(shapely_xy_point)
            self.gmsh_points.append(gmsh_point)
            self.points_labels.append(label)
        return gmsh_point

    def add_get_xy_segment(self, shapely_xy_point1, shapely_xy_point2, label):
        """Add a shapely segment (2-point line) to the gmsh model in the xy plane, or retrieve the existing gmsh segment with equivalent coordinates (within tol.).

        Args:
            shapely_xy_point1 (shapely.geometry.Point): first x, y coordinates
            shapely_xy_point2 (shapely.geometry.Point): second x, y coordinates
        """
        index, orientation = self.get_xy_segment_index_and_orientation(
            shapely_xy_point1, shapely_xy_point2
        )
        if index is not None:
            gmsh_segment = self.gmsh_xy_segments[index]
            self.xy_segments_secondary_labels[index] = label
        else:
            gmsh_segment = self.model.add_line(
                self.add_get_point(shapely_xy_point1),
                self.add_get_point(shapely_xy_point2),
            )
            self.shapely_xy_segments.append(
                shapely.geometry.LineString([shapely_xy_point1, shapely_xy_point2])
            )
            self.gmsh_xy_segments.append(gmsh_segment)
            self.xy_segments_main_labels.append(label)
            self.xy_segments_secondary_labels.append(None)
        return gmsh_segment, orientation

    def add_get_xy_line(self, shapely_xy_curve, label):
        """Add a shapely line (multi-point line) to the gmsh model in the xy plane, or retrieve the existing gmsh segment with equivalent coordinates (within tol.).

        Args:
            shapely_xy_curve (shapely.geometry.LineString): curve
        """
        segments = []
        for shapely_xy_point1, shapely_xy_point2 in zip(
            shapely_xy_curve.coords[:-1], shapely_xy_curve.coords[1:]
        ):
            gmsh_segment, orientation = self.add_get_xy_segment(
                Point(shapely_xy_point1), Point(shapely_xy_point2), label
            )
            if orientation:
                segments.append(gmsh_segment)
            else:
                segments.append(-gmsh_segment)
        self.model.add_physical(segments, f"{label}")

    def add_xy_surface(self, shapely_xy_polygon, label=None):
        """Add a xy surface corresponding to shapely_xy_polygon, or retrieve the existing gmsh model surface with equivalent coordinates (within tol.).

        Args:
            shapely_xy_polygon (shapely.geometry.Polygon):
        """
        # Create surface
        exterior_vertices = []
        hole_loops = []

        # Parse holes
        for polygon_hole in list(shapely_xy_polygon.interiors):
            hole_vertices = []
            for vertex in shapely.geometry.MultiPoint(polygon_hole.coords).geoms:
                # gmsh_point = self.add_get_point(vertex, label)
                hole_vertices.append(vertex)
            hole_loops.append(self.xy_channel_loop_from_vertices(hole_vertices, label))
        # Parse boundary
        for vertex in shapely.geometry.MultiPoint(
            shapely_xy_polygon.exterior.coords
        ).geoms:
            # gmsh_point = self.add_get_point(vertex, label)
            exterior_vertices.append(vertex)
        channel_loop = self.xy_channel_loop_from_vertices(exterior_vertices, label)

        # Create and log surface
        gmsh_surface = self.model.add_plane_surface(channel_loop, holes=hole_loops)
        self.gmsh_xy_surfaces.append(gmsh_surface)
        self.xy_surfaces_labels.append(label)
        return gmsh_surface


def break_line(line, other_line):
    intersections = line.intersection(other_line)
    if not intersections.is_empty:
        for intersection in (
            intersections.geoms if hasattr(intersections, "geoms") else [intersections]
        ):
            if intersection.type != "Point":
                new_coords_start, new_coords_end = intersection.boundary.geoms
                line = linemerge(split(line, new_coords_start))
                line = linemerge(split(line, new_coords_end))
    return line


def mesh_from_polygons(
    shapes_dict: OrderedDict,
    resolutions: Optional[Dict[str, float]] = None,
    default_resolution_min: float = 0.01,
    default_resolution_max: float = 0.5,
    filename: Optional[str] = None,
):

    import gmsh

    with pygmsh.occ.geometry.Geometry() as geometry:

        gmsh.initialize()

        # geometry = pygmsh.occ.geometry.Geometry()
        geometry.characteristic_length_min = default_resolution_min
        geometry.characteristic_length_max = default_resolution_max

        model = geometry.__enter__()

        # Break up shapes in order so that plane is tiled with non-overlapping layers
        shapes_tiled_dict = OrderedDict()
        for lower_index, (lower_name, lower_shape) in reversed(
            list(enumerate(shapes_dict.items()))
        ):
            diff_shape = lower_shape
            for higher_index, (higher_name, higher_shape) in reversed(
                list(enumerate(shapes_dict.items()))[:lower_index]
            ):
                diff_shape = diff_shape.difference(higher_shape)
            shapes_tiled_dict[lower_name] = diff_shape

        # Break up lines and polygon edges so that plane is tiled with no partially overlapping line segments
        polygons_broken_dict = OrderedDict()
        lines_broken_dict = OrderedDict()
        for first_index, (first_name, first_shape) in enumerate(shapes_dict.items()):
            first_shape = shapes_tiled_dict[first_name]
            broken_shapes = []
            for first_shape in (
                first_shape.geoms if hasattr(first_shape, "geoms") else [first_shape]
            ):
                # First line exterior
                first_exterior_line = (
                    LineString(first_shape.exterior)
                    if first_shape.type == "Polygon"
                    else first_shape
                )
                for second_index, (second_name, second_shapes) in enumerate(
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
                        for second_index, (second_name, second_shapes) in enumerate(
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
        meshtracker = MeshTracker(model=model)
        for line_name, line in lines_broken_dict.items():
            meshtracker.add_get_xy_line(line, line_name)

        # Add surfaces, reusing lines to simplify at early stage
        for polygon_name, polygon in polygons_broken_dict.items():
            meshtracker.add_xy_surface(polygon, polygon_name)

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

        # Fuse edges (bandaid)
        # gmsh.model.occ.synchronize()
        # gmsh.model.occ.removeAllDuplicates()
        # gmsh.model.occ.synchronize()

        # Extract all unique lines (TODO: identify interfaces in label)
        i = 0
        for index, line in enumerate(meshtracker.gmsh_xy_segments):
            model.add_physical(
                line,
                f"{meshtracker.xy_segments_main_labels[index]}_{meshtracker.xy_segments_secondary_labels[index]}_{i}",
            )
            i += 1

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
