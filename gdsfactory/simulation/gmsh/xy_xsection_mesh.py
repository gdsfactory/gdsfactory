from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from shapely.geometry import Polygon
from shapely.ops import unary_union

from gdsfactory.simulation.gmsh.mesh import mesh_from_polygons
from gdsfactory.simulation.gmsh.parse_component import (
    merge_by_material_func,
    process_buffers,
)
from gdsfactory.simulation.gmsh.parse_gds import cleanup_component
from gdsfactory.simulation.gmsh.parse_layerstack import (
    get_layers_at_z,
    order_layerstack,
)
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentOrReference


def xy_xsection_mesh(
    component: ComponentOrReference,
    z: float,
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
    round_tol: int = 4,
    simplify_tol: float = 1e-4,
    atol: Optional[float] = 1e-5,
):
    """Mesh xy cross-section of component at height z.

    Args:
        component (Component): gdsfactory component to mesh
        z (float): z-coordinate at which to sample the LayerStack
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
        atol: tolerance used to establish equivalency between vertices
    """
    # Fuse and cleanup polygons of same layer in case user overlapped them
    layer_polygons_dict = cleanup_component(
        component, layerstack, round_tol, simplify_tol
    )

    # GDS polygons to simulation polygons
    buffered_layer_polygons_dict, buffered_layerstack = process_buffers(
        layer_polygons_dict, layerstack
    )

    # Find layers present at this z-level
    layers = get_layers_at_z(buffered_layerstack, z)

    # Remove terminal layers and merge polygons
    layer_order = order_layerstack(buffered_layerstack)  # gds layers
    ordered_layers = [value for value in layer_order if value in set(layers)]
    shapes = OrderedDict() if extra_shapes_dict is None else extra_shapes_dict
    for layername in ordered_layers:
        for simulation_layername, (
            gds_layername,
            _next_simulation_layername,
            this_layer_polygons,
            _next_layer_polygons,
        ) in buffered_layer_polygons_dict.items():
            if simulation_layername == layername:
                shapes[gds_layername] = this_layer_polygons

    # Add background polygon
    if background_tag is not None:
        bounds = unary_union(list(shapes.values())).bounds
        shapes[background_tag] = Polygon(
            [
                [bounds[0] - background_padding[0], bounds[1] - background_padding[1]],
                [bounds[0] - background_padding[0], bounds[3] + background_padding[3]],
                [bounds[2] + background_padding[2], bounds[3] + background_padding[3]],
                [bounds[2] + background_padding[2], bounds[1] - background_padding[1]],
            ]
        )

    # Merge by material
    if merge_by_material:
        shapes = merge_by_material_func(shapes, layerstack)

    # Mesh
    return mesh_from_polygons(
        shapes,
        resolutions=resolutions,
        mesh_scaling_factor=mesh_scaling_factor,
        filename=filename,
        default_resolution_min=default_resolution_min,
        default_resolution_max=default_resolution_max,
        global_meshsize_array=global_meshsize_array,
        global_meshsize_interpolant_func=global_meshsize_interpolant_func,
        atol=atol,
    )


if __name__ == "__main__":
    import gdsfactory as gf

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

    from gdsfactory.pdk import get_layer_stack

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                "slab90",
                "core",
                "via_contact",
                "undercut",
                "box",
                "substrate",
                # "clad",
                "metal1",
            )
        }
    )

    resolutions = {}
    resolutions["core"] = {"resolution": 0.05, "distance": 0.1}
    resolutions["via_contact"] = {"resolution": 0.1, "distance": 0}

    geometry = xy_xsection_mesh(
        component=c,
        z=-6,
        layerstack=filtered_layerstack,
        resolutions=resolutions,
        # background_tag="Oxide",
        filename="mesh.msh",
    )
    # print(geometry)

    # import gmsh

    # gmsh.write("mesh.msh")
    # gmsh.clear()
    # geometry.__exit__()

    import meshio

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

    # # for layer, polygons in heaters.get_polygons(by_spec=True).items():
    # #     print(layer, polygons)
