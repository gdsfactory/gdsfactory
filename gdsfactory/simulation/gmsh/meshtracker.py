from __future__ import annotations

from collections import OrderedDict

import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, split
import numpy as np


class MeshTracker:
    def __init__(self, model, atol=1e-3):
        """Map between shapely and gmsh.

        Shapely is useful for built-in geometry equivalencies and extracting orientation, instead of doing it manually
        We can also track information about the entities using labels (useful for selective mesh refinement later)

        TODO: Break into subclasses
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

    def get_point_index(self, xy_point, z):
        return next(
            (
                index
                for index, (shapely_point, stored_z) in enumerate(self.shapely_points)
                if xy_point.equals_exact(shapely_point, self.atol) and z == stored_z
            ),
            None,
        )

    def get_xy_segment_index_and_orientation(self, xy_point1, xy_point2, z1=0, z2=0):
        """Note: orientation of z1 <--> z2 not accounted (occ kernel does not need)."""
        xy_line = shapely.geometry.LineString([xy_point1, xy_point2])
        for index, (shapely_line, _stored_z1, _stored_z2) in enumerate(
            self.shapely_xy_segments
        ):
            if xy_line.equals(shapely_line):
                first_xy_line, last_xy_line = xy_line.boundary.geoms
                first_xy, last_xy = shapely_line.boundary.geoms
                return (
                    (index, True) if first_xy_line.equals(first_xy) else (index, False)
                )
        return None, 1

    def get_gmsh_points_from_label(self, label):
        indices = [
            idx for idx, value in enumerate(self.points_labels) if value == label
        ]
        return [self.gmsh_points[index]._id for index in indices]

    def get_gmsh_xy_lines_from_label(self, label):
        indices = [
            idx
            for idx, value in enumerate(self.xy_segments_main_labels)
            if value == label
        ]
        return [self.gmsh_xy_segments[index]._id for index in indices]

    def get_gmsh_xy_surfaces_from_label(self, label):
        indices = [
            idx for idx, value in enumerate(self.xy_surfaces_labels) if value == label
        ]
        return [self.gmsh_xy_surfaces[index]._id for index in indices]

    """Channel loop utils (no need to track)"""

    def xy_channel_loop_from_vertices(self, vertices, label):
        edges = []
        for vertex1, vertex2 in [
            (vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)
        ]:
            gmsh_line, orientation = self.add_get_xy_segment(vertex1, vertex2, label)
            if orientation:
                edges.append(gmsh_line)
            else:
                edges.append(-gmsh_line)
        return self.model.add_curve_loop(edges)

    """Adding geometry"""

    def add_get_point(self, shapely_xy_point, z=0, label=None):
        """Add a shapely point to the gmsh model, or retrieve the existing gmsh model points with equivalent coordinates (within tol.).

        Args:
            shapely_xy_point (shapely.geometry.Point): x, y coordinates
            z: float, z-coordinate
        """
        index = self.get_point_index(shapely_xy_point, z)
        if index is not None:
            gmsh_point = self.gmsh_points[index]
        else:
            gmsh_point = self.model.add_point(
                [shapely_xy_point.x, shapely_xy_point.y, z]
            )
            self.shapely_points.append((shapely_xy_point, z))
            self.gmsh_points.append(gmsh_point)
            self.points_labels.append(label)
        return gmsh_point

    def add_get_xy_segment(
        self, shapely_xy_point1, shapely_xy_point2, label, z1=0, z2=0
    ):
        """Add a shapely segment (2-point line) to the gmsh model, or retrieve the existing gmsh segment with equivalent coordinates (within tol.).

        Args:
            shapely_xy_point1 (shapely.geometry.Point): first x, y coordinates
            z1: float, first z-coordinate
            shapely_xy_point2 (shapely.geometry.Point): second x, y coordinates
            z2: float, second z-coordinate
        """
        index, orientation = self.get_xy_segment_index_and_orientation(
            shapely_xy_point1, shapely_xy_point2, z1, z2
        )
        if index is not None:
            gmsh_segment = self.gmsh_xy_segments[index]
            self.xy_segments_secondary_labels[index] = label
        else:
            gmsh_segment = self.model.add_line(
                self.add_get_point(shapely_xy_point1, z1),
                self.add_get_point(shapely_xy_point2, z2),
            )
            self.shapely_xy_segments.append(
                (
                    shapely.geometry.LineString([shapely_xy_point1, shapely_xy_point2]),
                    z1,
                    z2,
                )
            )
            self.gmsh_xy_segments.append(gmsh_segment)
            self.xy_segments_main_labels.append(label)
            self.xy_segments_secondary_labels.append(None)
        return gmsh_segment, orientation

    def add_get_xy_line(self, shapely_xy_curve, label, zs=None):
        """Add a shapely line (multi-point line) to the gmsh model in the xy plane, or retrieve the existing gmsh segment with equivalent coordinates (within tol.).

        Args:
            shapely_xy_curve (shapely.geometry.LineString): curve
        """
        segments = []
        zs = zs or [0] * len(shapely_xy_curve.coords)
        for shapely_xy_point1, shapely_xy_point2, z1, z2 in zip(
            shapely_xy_curve.coords[:-1], shapely_xy_curve.coords[1:], zs[:-1], zs[1:]
        ):
            gmsh_segment, orientation = self.add_get_xy_segment(
                Point(shapely_xy_point1), Point(shapely_xy_point2), label, z1, z2
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
        hole_loops = []

        # Parse holes
        for polygon_hole in list(shapely_xy_polygon.interiors):
            hole_vertices = list(shapely.geometry.MultiPoint(polygon_hole.coords).geoms)
            hole_loops.append(self.xy_channel_loop_from_vertices(hole_vertices, label))
        exterior_vertices = list(
            shapely.geometry.MultiPoint(shapely_xy_polygon.exterior.coords).geoms
        )

        channel_loop = self.xy_channel_loop_from_vertices(exterior_vertices, label)

        # Create and log surface
        gmsh_surface = self.model.add_plane_surface(channel_loop, holes=hole_loops)
        self.gmsh_xy_surfaces.append(gmsh_surface)
        self.xy_surfaces_labels.append(label)
        return gmsh_surface


def break_line(line, other_line):
    initial_settings = np.seterr()
    np.seterr(invalid="ignore")
    intersections = line.intersection(other_line)
    np.seterr(**initial_settings)
    if not intersections.is_empty:
        for intersection in (
            intersections.geoms if hasattr(intersections, "geoms") else [intersections]
        ):
            if intersection.type != "Point":
                new_coords_start, new_coords_end = intersection.boundary.geoms
                line = linemerge(split(line, new_coords_start))
                line = linemerge(split(line, new_coords_end))
    return line


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

    from gdsfactory.simulation.gmsh import mesh_from_polygons

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
