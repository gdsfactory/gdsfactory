from collections import OrderedDict

from gdsfactory.simulation.gmsh.mesh import mesh_from_polygons
from gdsfactory.simulation.gmsh.parse_gds import fuse_component_layer
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
):

    # Find layers present at this z-level
    layers = get_layers_at_z(layerstack, z)

    # Fuse and cleanup polygons of same layer in case user overlapped them
    layer_dict = layerstack.to_dict()
    filtered_layerdict = {key: layer_dict[key] for key in layers}
    layer_polygons_dict = {}
    for layername, layer in filtered_layerdict.items():
        layer_polygons_dict[layername] = fuse_component_layer(
            component, layername, layer
        )

    # Reorder polygons according to meshorder
    layer_order = order_layerstack(layerstack)
    ordered_layers = [value for value in layer_order if value in set(layer_dict.keys())]
    shapes = OrderedDict()
    for layer in ordered_layers:
        shapes[layer] = layer_polygons_dict[layer]

    # for key, item in shapes.items():
    #     print(key, item)

    # Mesh
    return mesh_from_polygons(shapes, filename="mesh.msh")


if __name__ == "__main__":

    import gdsfactory as gf

    # T  = gf.Component()
    waveguide = gf.components.straight_pin(length=10, taper=None)  # .extract((3,0))
    waveguide.show()

    from gdsfactory.tech import get_layer_stack_generic

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
                "via_contact",
            )  # "slab90", "via_contact")#"via_contact") # "slab90", "core"
        }
    )

    geometry = xy_xsection_mesh(
        component=waveguide,
        z=1,
        layerstack=filtered_layerstack,
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

    # # line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    # # meshio.write("facet_mesh.xdmf", line_mesh)

    # # triangle_mesh = create_mesh(mesh_from_file, "tetra", prune_z=False)
    # # meshio.write("mesh.xdmf", triangle_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("mesh.xdmf", triangle_mesh)

    # # for layer, polygons in heaters.get_polygons(by_spec=True).items():
    # #     print(layer, polygons)
