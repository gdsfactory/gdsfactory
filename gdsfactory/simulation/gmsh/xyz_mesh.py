from __future__ import annotations

from typing import Dict, Optional

import gmsh

from gdsfactory.simulation.gmsh.parse_component import buffers_to_lists
from gdsfactory.simulation.gmsh.parse_gds import cleanup_component
from gdsfactory.simulation.gmsh.parse_layerstack import order_layerstack
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentOrReference


def add_get_point(occ, x, y, z, points_dict):
    """Add a point to the gmsh model, or retrieve a previously-defined point.

    Args:
        model: GMSH model
        x: float, x-coordinate
        y: float, y-coordinate
        z: float, z-coordinate
    """
    if (x, y, z) not in points_dict.keys():
        points_dict[(x, y, z)] = occ.add_point(x, y, z)
    return points_dict[(x, y, z)]


def add_get_segment(occ, xyz1, xyz2, lines_dict, points_dict):
    """Add a segment (2-point line) to the gmsh model, or retrieve a previously-defined segment.

    Note that the OCC kernel does not care about orientation.

    Args:
        model: GMSH model
        xyz1: first [x,y,z] coordinate
        xyz2: second [x,y,z] coordinate
    """
    if (xyz1, xyz2) in lines_dict.keys():
        return lines_dict[(xyz1, xyz2)]
    elif (xyz2, xyz1) in lines_dict.keys():
        return lines_dict[(xyz2, xyz1)]
    else:
        lines_dict[(xyz1, xyz2)] = occ.add_line(
            add_get_point(occ, xyz1[0], xyz1[1], xyz1[2], points_dict),
            add_get_point(occ, xyz2[0], xyz2[1], xyz2[2], points_dict),
        )
        return lines_dict[(xyz1, xyz2)]


def channel_loop_from_vertices(occ, vertices, lines_dict, points_dict):
    """Add a curve loop from the list of vertices.

    Args:
        model: GMSH model
        vertices: list of [x,y,z] coordinates
    """
    edges = []
    for vertex1, vertex2 in [
        (vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)
    ]:
        gmsh_line = add_get_segment(occ, vertex1, vertex2, lines_dict, points_dict)
        edges.append(gmsh_line)
    return occ.add_curve_loop(edges)


def add_surface(occ, vertices, lines_dict, points_dict):
    """Add a surface composed of the segments formed by vertices.

    Args:
        vertices: List of xyz coordinates, whose subsequent entries define a closed loop.
    """
    channel_loop = channel_loop_from_vertices(occ, vertices, lines_dict, points_dict)
    return occ.add_plane_surface([channel_loop])


def add_volume(occ, entry, lines_dict, points_dict, exterior=True, interior_index=0):
    """Create shape from a list of the same buffered polygon and a list of z-values.

    Args:
        polygons: shapely polygons from the GDS
        zs: list of z-values for each polygon

    Returns:
        GMSH volume for this entry.
    """
    # Draw bottom surface
    bottom_polygon = entry[0][1]
    bottom_polygon_z = entry[0][0]
    if exterior:
        bottom_polygon_vertices = [
            (x, y, bottom_polygon_z) for x, y in bottom_polygon.exterior.coords
        ]
    else:
        bottom_polygon_vertices = [
            (x, y, bottom_polygon_z)
            for x, y in bottom_polygon.interiors[interior_index].coords
        ]
    gmsh_surfaces = [add_surface(occ, bottom_polygon_vertices, lines_dict, points_dict)]
    # Draw top surface
    top_polygon = entry[-1][1]
    top_polygon_z = entry[-1][0]
    if exterior:
        top_polygon_vertices = [
            (x, y, top_polygon_z) for x, y in top_polygon.exterior.coords
        ]
    else:
        top_polygon_vertices = [
            (x, y, top_polygon_z)
            for x, y in top_polygon.interiors[interior_index].coords
        ]
    gmsh_surfaces.append(
        add_surface(occ, top_polygon_vertices, lines_dict, points_dict)
    )
    # Draw vertical surfaces
    for pair_index in range(len(entry) - 1):
        if exterior:
            bottom_polygon = entry[pair_index][1].exterior.coords
            top_polygon = entry[pair_index + 1][1].exterior.coords
        else:
            bottom_polygon = entry[pair_index][1].interiors[interior_index].coords
            top_polygon = entry[pair_index + 1][1].interiors[interior_index].coords
        bottom_z = entry[pair_index][0]
        top_z = entry[pair_index + 1][0]
        for facet_pt_ind in range(len(bottom_polygon) - 1):
            facet_pt1 = (
                bottom_polygon[facet_pt_ind][0],
                bottom_polygon[facet_pt_ind][1],
                bottom_z,
            )
            facet_pt2 = (
                bottom_polygon[facet_pt_ind + 1][0],
                bottom_polygon[facet_pt_ind + 1][1],
                bottom_z,
            )
            facet_pt3 = (
                top_polygon[facet_pt_ind + 1][0],
                top_polygon[facet_pt_ind + 1][1],
                top_z,
            )
            facet_pt4 = (
                top_polygon[facet_pt_ind][0],
                top_polygon[facet_pt_ind][1],
                top_z,
            )
            facet_vertices = [facet_pt1, facet_pt2, facet_pt3, facet_pt4, facet_pt1]
            gmsh_surfaces.append(
                add_surface(occ, facet_vertices, lines_dict, points_dict)
            )

    # Return volume from closed shell
    surface_loop = occ.add_surface_loop(gmsh_surfaces)
    return occ.add_volume([surface_loop])


def add_volume_with_holes(occ, entry, lines_dict, points_dict):
    """Returns volume, removing intersection with hole volumes."""
    exterior = add_volume(occ, entry, lines_dict, points_dict, exterior=True)
    interiors = [
        add_volume(
            occ,
            entry,
            lines_dict,
            points_dict,
            exterior=False,
            interior_index=interior_index,
        )
        for interior_index in range(len(entry[0][1].interiors))
    ]
    if interiors:
        for interior in interiors:
            exterior = occ.cut(
                [(3, exterior)], [(3, interior)], removeObject=True, removeTool=True
            )
            occ.synchronize()
            exterior = exterior[0][0][1]  # Parse `outDimTags', `outDimTagsMap'
    return exterior


def create_shapes(occ, buffered_layer_polygons_dict):
    """Loop over layers and polygons to create base shapes.

    Args:
        buffered_layer_polygons_dict:

    Returns:
        Dict of layernames: occ volume ids
    """
    shapes = {}
    lines_dict = {}
    points_dict = {}
    for layername, entry in buffered_layer_polygons_dict.items():
        subshapes_list = []
        subshapes_list = [
            add_volume_with_holes(occ, polygon, lines_dict, points_dict)
            for polygon in entry
        ]
        shapes[layername] = subshapes_list

    return shapes


def xyz_mesh(
    component: ComponentOrReference,
    layerstack: LayerStack,
    resolutions: Optional[Dict] = None,
    default_resolution_min: float = 0.01,
    default_resolution_max: float = 0.5,
    filename: Optional[str] = None,
    verbosity: Optional[bool] = False,
    override_volumes: Optional[Dict] = None,
    round_tol: int = 3,
    simplify_tol: float = 1e-2,
):
    """Full 3D mesh of component.

    Args:
        component (Component): gdsfactory component to mesh
        layerstack (LayerStack): gdsfactory LayerStack to parse
        resolutions (Dict): Pairs {"layername": {"resolution": float, "distance": "float}} to roughly control mesh refinement
        default_resolution_min (float): gmsh minimal edge length
        default_resolution_max (float): gmsh maximal edge length
        filename (str, path): where to save the .msh file
        override_volumes: Dict of {physical: [volume_ids]}. If not None, will manually assign physicals to the volume IDs (after performing coherence), deleting extra volumes
        round_tol: during gds --> mesh conversion cleanup, number of decimal points at which to round the gdsfactory/shapely points before introducing to gmsh
        simplify_tol: during gds --> mesh conversion cleanup, shapely "simplify" tolerance (make it so all points are at least separated by this amount)
    """
    # Fuse and cleanup polygons of same layer in case user overlapped them
    layer_polygons_dict = cleanup_component(
        component, layerstack, round_tol, simplify_tol
    )

    # GDS polygons to simulation polygons
    buffered_layer_polygons_dict = buffers_to_lists(layer_polygons_dict, layerstack)

    occ = gmsh.model.occ
    gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)

    shapes = create_shapes(occ, buffered_layer_polygons_dict)
    occ.synchronize()

    # Iterate through objects, removing overlaps
    # Don't remove entities at this stage
    broken_shapes = {}
    ordered_layers = order_layerstack(layerstack)  # gds layers
    for index, layername in enumerate(ordered_layers):
        if index == 0:
            broken_shapes[layername] = shapes[layername]
            tool = shapes[layername]
        else:
            new_obj = []
            for obj_id in shapes[layername]:
                new_tool = []
                for tool_id in tool:
                    new_obj.append(
                        [
                            x[1]
                            for x in occ.cut(
                                [(3, obj_id)],
                                [(3, tool_id)],
                                removeObject=False,
                                removeTool=False,
                            )[0]
                        ]
                        or [obj_id]
                    )
                    new_tool.append(
                        [
                            x[1]
                            for x in occ.fuse(
                                [(3, obj_id)],
                                [(3, tool_id)],
                                removeObject=False,
                                removeTool=False,
                            )[0]
                        ]
                    )
            tool = [item for sublist in new_tool for item in sublist]
            broken_shapes[layername] = [
                item for sublist in new_obj for item in sublist if item
            ]

    # Remove redundant entities
    occ.synchronize()
    current_entities = [item for sublist in broken_shapes.values() for item in sublist]
    entities_to_remove = [
        (dimension, entity)
        for dimension, entity in occ.getEntities(dim=3)
        if entity not in current_entities
    ]
    occ.remove(entities_to_remove, recursive=False)
    occ.synchronize()

    # Fix remaining interfaces
    # HACK: Assuming GMSH iterates in entity order, reassing entities by relative order if they changed after duplicate deletion
    # Might be more robust if we re-identify with something calculated from the entity like center and bounding box
    pre_removeAllDuplicates = {entity for dimension, entity in occ.getEntities(dim=3)}
    occ.removeAllDuplicates()
    occ.synchronize()
    post_removeAllDuplicates = {entity for dimension, entity in occ.getEntities(dim=3)}

    if override_volumes:
        for layername, volume_ids in override_volumes.items():
            gmsh.model.addPhysicalGroup(3, volume_ids, name=layername)
        current_entities = [
            item for sublist in override_volumes.values() for item in sublist
        ]
        entities_to_remove = [
            (dimension, entity)
            for dimension, entity in occ.getEntities(dim=3)
            if entity not in current_entities
        ]
        occ.remove(entities_to_remove, recursive=False)
        occ.synchronize()

    else:  # try to smartly reassign volumes
        old_entities = list(pre_removeAllDuplicates - post_removeAllDuplicates)
        new_entities = list(post_removeAllDuplicates - pre_removeAllDuplicates)

        for layername, layer_old_entities in broken_shapes.items():
            layer_new_entities = []
            for entity in layer_old_entities:
                if entity in old_entities:
                    layer_new_entities.append(new_entities[old_entities.index(entity)])
                else:
                    layer_new_entities.append(entity)
            broken_shapes[layername] = layer_new_entities

        for layername, entities in broken_shapes.items():
            gmsh.model.addPhysicalGroup(3, entities, name=layername)

    # Refine
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
            "RegionsList",
            broken_shapes[label],
        )
        refinement_fields.extend((n + 1,))
        n += 2

    # Use the smallest element size overall
    gmsh.model.mesh.field.add("Min", n)
    gmsh.model.mesh.field.setNumbers(n, "FieldsList", refinement_fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(n)

    # Turn off default meshing options
    gmsh.model.mesh.MeshSizeFromPoints = 0
    gmsh.model.mesh.MeshSizeFromCurvature = 0
    gmsh.model.mesh.MeshSizeExtendFromBoundary = 0

    occ.synchronize()

    # model.generate_mesh(3)
    gmsh.option.setNumber(
        "General.Terminal", 1 if verbosity else 0
    )  # 1 verbose, 0 otherwise
    gmsh.model.mesh.generate(3)
    gmsh.write(filename)

    # Mesh
    gmsh.finalize()
    return True


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.component.Component()
    # waveguide = c << gf.get_component(gf.components.straight_pin(length=10, taper=None))
    # undercut = c << gf.get_component(
    #     gf.components.rectangle(
    #         size=(5.0, 5.0),
    #         layer="UNDERCUT",
    #         centered=True,
    #     )
    # ).move(destination=[4, 0])
    # c.show()

    c = gf.component.Component()
    # waveguide = c << gf.get_component(gf.components.straight_pin(length=5, taper=None))
    ring = c << gf.get_component(gf.components.ring_crow)
    # bend = c << gf.get_component(gf.components.spiral_double(cross_section="rib"))
    # c = gf.components.spiral_racetrack(cross_section='rib')
    c.plot()
    # undercut = c << gf.get_component(
    #     gf.components.rectangle(
    #         size=(5.0, 5.0),
    #         layer="UNDERCUT",
    #         centered=True,
    #     )
    # ).move(destination=[4, 0])
    c.show()

    from gdsfactory.pdk import get_layer_stack

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                # "slab90",
                "core",
                # "via_contact",
                # "undercut",
                # "box",
                # "substrate",
                # "clad",
                # "metal1",
            )
        }
    )

    # filtered_layerstack.layers["via_contact"].info["mesh_order"] = 4
    # filtered_layerstack.layers["clad"].info["mesh_order"] = 5

    resolutions = {
        "core": {"resolution": 0.1},
        # "slab90": {"resolution": 0.4},
        # "via_contact": {"resolution": 0.4},
    }
    geometry = xyz_mesh(
        component=c,
        layerstack=filtered_layerstack,
        resolutions=resolutions,
        filename="mesh.msh",
        verbosity=False,
    )
    # print(geometry)

    # import gmsh

    # gmsh.write("mesh.msh")
    # gmsh.clear()
    # geometry.__exit__()

    # import meshio

    # mesh_from_file = meshio.read("mesh.msh")

    # def create_mesh(mesh, cell_type, prune_z=False):
    #     cells = mesh.get_cells_type(cell_type)
    #     cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    #     points = mesh.points
    #     return meshio.Mesh(
    #         points=points,
    #         cells={cell_type: cells},
    #         cell_data={"name_to_read": [cell_data]},
    #     )

    # line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    # meshio.write("facet_mesh.xdmf", line_mesh)

    # triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    # meshio.write("mesh.xdmf", triangle_mesh)

    # # for layer, polygons in heaters.get_polygons(by_spec=True).items():
    # #     print(layer, polygons)
