from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

from shapely.geometry import Polygon
from shapely.ops import unary_union

from gdsfactory.simulation.gmsh.mesh import mesh_from_polygons
from gdsfactory.simulation.gmsh.parse_gds import cleanup_component
from gdsfactory.simulation.gmsh.parse_layerstack import (
    get_layers_at_z,
    order_layerstack,
)
from gdsfactory.tech import LayerStack
from gdsfactory.types import ComponentOrReference


def xy_xsection_mesh(
    component: ComponentOrReference,
    z: float,
    layerstack: LayerStack,
    resolutions: Optional[Dict],
    default_resolution_min: float = 0.01,
    default_resolution_max: float = 0.5,
    background_tag: Optional[str] = None,
    background_padding: Tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0),
    filename: Optional[str] = None,
):
    """Mesh xy cross-section of component at height z.

    Args:
        component (Component): gdsfactory component to mesh
        z (float): z-coordinate at which to sample the LayerStack
        layerstack (LayerStack): gdsfactory LayerStack to parse
        resolutions (Dict): Pairs {"layername": {"resolution": float, "distance": "float}} to roughly control mesh refinement
        default_resolution_min (float): gmsh minimal edge length
        default_resolution_max (float): gmsh maximal edge length
        background_tag (str): name of the background layer to add (default: no background added)
        background_padding (Tuple): [xleft, ydown, xright, yup] distances to add to the components and to fill with background_tag
    """
    # Find layers present at this z-level
    layers = get_layers_at_z(layerstack, z)

    # Fuse and cleanup polygons of same layer in case user overlapped them
    layer_polygons_dict = cleanup_component(component, layerstack)

    # Reorder polygons according to meshorder
    layer_order = order_layerstack(layerstack)
    ordered_layers = [value for value in layer_order if value in set(layers)]
    shapes = OrderedDict()
    for layer in ordered_layers:
        shapes[layer] = layer_polygons_dict[layer]

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

    # Mesh
    return mesh_from_polygons(
        shapes,
        resolutions=resolutions,
        filename="mesh.msh",
        default_resolution_min=default_resolution_min,
        default_resolution_max=default_resolution_max,
    )


if __name__ == "__main__":

    import gdsfactory as gf

    # T  = gf.Component()
    waveguide = gf.components.straight_pin(length=10, taper=None)
    waveguide.show()

    from gdsfactory.tech import get_layer_stack_generic

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
                "via_contact",
            )
        }
    )

    resolutions = {}
    resolutions["core"] = {"resolution": 0.05, "distance": 0.1}
    resolutions["via_contact"] = {"resolution": 0.1, "distance": 0}

    geometry = xy_xsection_mesh(
        component=waveguide,
        z=0.09,
        layerstack=filtered_layerstack,
        resolutions=resolutions,
        background_tag="Oxide",
    )
    # print(geometry)

    # import gmsh

    # gmsh.write("mesh.msh")
    # gmsh.clear()
    # geometry.__exit__()

    import meshio

    mesh_from_file = meshio.read("mesh.msh")

    def create_mesh(mesh, cell_type, prune_z=False):
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

    # # for layer, polygons in heaters.get_polygons(by_spec=True).items():
    # #     print(layer, polygons)
