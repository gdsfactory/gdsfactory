import gmsh


def global_callback_refinement(
    model,
    global_meshsize_array,
    global_meshsize_interpolant_func,
):
    """Refine gmsh/pygmsh model according to a global callback dict.

    Args:
        model
        global_meshsize_array
        global_meshsize_interpolant_func
    Returns:
        model
    """
    global_meshsize_interpolant = global_meshsize_interpolant_func(
        global_meshsize_array[:, 0:3], global_meshsize_array[:, 3]
    )

    def meshSizeCallback(dim, tag, x, y, z, lc):
        return min(lc, global_meshsize_interpolant(x, y, z))

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)

    # Turn off default meshing options
    gmsh.model.mesh.MeshSizeFromPoints = 0
    gmsh.model.mesh.MeshSizeFromCurvature = 0
    gmsh.model.mesh.MeshSizeExtendFromBoundary = 0

    return model


def surface_interface_refinement(
    model,
    meshtracker,
    resolutions,
    default_resolution_min,
    default_resolution_max,
):
    """Refine gmsh/pygmsh model according to a resolutions dict.

    Args:
        model
        meshtracker
        resolutions
        default_resolution_min
        default_resolution_max
    Returns:
        model
        meshtracker
    """
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
        refinement_fields.extend((n + 1, n + 3))
        n += 4

    # Use the smallest element size overall
    gmsh.model.mesh.field.add("Min", n)
    gmsh.model.mesh.field.setNumbers(n, "FieldsList", refinement_fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(n)

    # Turn off default meshing options
    gmsh.model.mesh.MeshSizeFromPoints = 0
    gmsh.model.mesh.MeshSizeFromCurvature = 0
    gmsh.model.mesh.MeshSizeExtendFromBoundary = 0

    return model, meshtracker
