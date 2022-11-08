import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import linemerge, split


def round_coordinates(geom, ndigits=3):
    """Round coordinates to n_digits to eliminate floating point errors."""

    def _round_coords(x, y, z=None):
        x = round(x, ndigits)
        y = round(y, ndigits)

        if z is not None:
            z = round(x, ndigits)

        return [c for c in (x, y, z) if c is not None]

    return shapely.ops.transform(_round_coords, geom)


def fuse_component_layer(component, layername, layer, round_tol=2, simplify_tol=1e-2):
    """Take all polygons from a layer, and returns a single (Multi)Polygon shapely object."""
    layer_component = component.extract(layer["layer"])
    shapely_polygons = []
    for polygon in layer_component.get_polygons():
        shapely_polygons.append(
            round_coordinates(shapely.geometry.Polygon(polygon), round_tol)
        )
    fused_polygons = shapely.ops.unary_union(shapely_polygons).simplify(
        simplify_tol, preserve_topology=True
    )
    return fused_polygons


def to_polygons(geometries):
    for geometry in geometries:
        if isinstance(geometry, Polygon):
            yield geometry
        else:
            yield from geometry


def to_lines(geometries):
    for geometry in geometries:
        if isinstance(geometry, LineString):
            yield geometry
        else:
            yield from geometry


def tile_shapes(shapes_dict):
    """Break up shapes in order so that plane is tiled with non-overlapping layers."""
    shapes_tiled_dict = {}
    for lower_index, (lower_name, lower_shapes) in reversed(
        list(enumerate(shapes_dict.items()))
    ):
        tiled_lower_shapes = []
        for lower_shape in (
            lower_shapes.geoms if hasattr(lower_shapes, "geoms") else [lower_shapes]
        ):
            diff_shape = lower_shape
            for _higher_index, (_higher_name, higher_shapes) in reversed(
                list(enumerate(shapes_dict.items()))[:lower_index]
            ):
                for higher_shape in (
                    higher_shapes.geoms
                    if hasattr(higher_shapes, "geoms")
                    else [higher_shapes]
                ):
                    diff_shape = diff_shape.difference(higher_shape)
            tiled_lower_shapes.append(diff_shape)
        if lower_shape.type == "Polygon" or lower_shape.type == "MultiPolygon":
            shapes_tiled_dict[lower_name] = MultiPolygon(
                to_polygons(tiled_lower_shapes)
            )
        else:
            shapes_tiled_dict[lower_name] = MultiPolygon(to_lines(tiled_lower_shapes))

    return shapes_tiled_dict


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
