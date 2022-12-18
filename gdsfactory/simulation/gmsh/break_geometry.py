"""Like Gmsh OCC kernel BooleanFragments, but (1) uses a meshorder to avoid generation of new surfaces, which (2) allows keeping track of physicals."""
from collections import OrderedDict

from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import linemerge, split

from gdsfactory.simulation.gmsh.parse_gds import tile_shapes


def break_line(line, other_line):
    intersections = line.intersection(other_line)
    if not intersections.is_empty:
        for intersection in (
            intersections.geoms if hasattr(intersections, "geoms") else [intersections]
        ):
            if intersection.geom_type == "Point":
                line = linemerge(split(line, intersection))
            else:
                new_coords_start, new_coords_end = intersection.boundary.geoms
                line = linemerge(split(line, new_coords_start))
                line = linemerge(split(line, new_coords_end))

    return line


def break_geometry(shapes_dict: OrderedDict):
    """Break up lines and polygon edges so that plane is tiled with no partially overlapping line segments.

    TODO: breakup in smaller functions.

    Args:
        shapes_dict: arbitrary dict of shapely polygons and lines, with ordering setting mesh priority

    Returns:
        polygons_broken_dict: dict of shapely polygons, with smallest number of individual vertices
        lines_broken_dict: dict of shapely lines, with smallest number of individual vertices
    """
    # Break up shapes in order so that plane is tiled with non-overlapping layers
    shapes_tiled_dict = tile_shapes(shapes_dict)

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
                if first_shape.geom_type == "Polygon"
                else first_shape
            )
            for second_name, second_shapes in shapes_dict.items():
                # Do not compare to itself
                if second_name != first_name:
                    second_shapes = shapes_tiled_dict[second_name]
                    for second_shape in (
                        second_shapes.geoms
                        if hasattr(second_shapes, "geoms")
                        else [second_shapes]
                    ):
                        # Second line exterior
                        second_exterior_line = (
                            LineString(second_shape.exterior)
                            if second_shape.geom_type == "Polygon"
                            else second_shape
                        )
                        first_exterior_line = break_line(
                            first_exterior_line, second_exterior_line
                        )
                        # Second line interiors
                        for second_interior_line in (
                            second_shape.interiors
                            if second_shape.geom_type == "Polygon"
                            else []
                        ):
                            second_interior_line = LineString(second_interior_line)
                            first_exterior_line = break_line(
                                first_exterior_line, second_interior_line
                            )
            # First line interiors
            if first_shape.geom_type in ["Polygon", "MultiPolygon"]:
                first_shape_interiors = []
                for first_interior_line in first_shape.interiors:
                    first_interior_line = LineString(first_interior_line)
                    for second_name, second_shapes in shapes_dict.items():
                        if second_name != first_name:
                            second_shapes = shapes_tiled_dict[second_name]
                            for second_shape in (
                                second_shapes.geoms
                                if hasattr(second_shapes, "geoms")
                                else [second_shapes]
                            ):
                                # Exterior
                                second_exterior_line = (
                                    LineString(second_shape.exterior)
                                    if second_shape.geom_type == "Polygon"
                                    else second_shape
                                )
                                first_interior_line = break_line(
                                    first_interior_line, second_exterior_line
                                )
                                # Interiors
                                for second_interior_line in (
                                    second_shape.interiors
                                    if second_shape.geom_type == "Polygon"
                                    else []
                                ):
                                    second_interior_line = LineString(
                                        second_interior_line
                                    )
                                    first_interior_line = break_line(
                                        first_interior_line, second_interior_line
                                    )
                    first_shape_interiors.append(first_interior_line)
            if first_shape.geom_type in ["Polygon", "MultiPolygon"]:
                broken_shapes.append(
                    Polygon(first_exterior_line, holes=first_shape_interiors)
                )
            else:
                broken_shapes.append(LineString(first_exterior_line))
        if broken_shapes:
            if first_shape.geom_type in ["Polygon", "MultiPolygon"]:
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

    return polygons_broken_dict, lines_broken_dict
