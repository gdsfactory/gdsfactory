from typing import Dict, Optional, Tuple

import pygmsh

from gdsfactory.pdk import get_layer_stack
from gdsfactory.tech import LayerStack
from gdsfactory.types import ComponentOrReference, Layer

import numpy as np


def surface_loop_from_vertices(model, xmin, xmax, ymin, ymax, zmin, zmax, resolution):
    """Returns surface loop of prism from bounding box.

    Args:
        xmin: minimal x-value for bounding box
        xmax: maximal x-value for bounding box
        ymin: minimal y-value for bounding box
        ymax: maximal y-value for bounding box
        zmin: minimal z-value for bounding box
        zmax: maximal z-value for bounding box

    """
    channel_surfaces = []
    for coords in [
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmax],
            [xmin, ymax, zmin],
        ],
        [
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmax, ymax, zmin],
        ],
        [
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmin, ymin, zmax],
            [xmin, ymin, zmin],
        ],
        [
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
            [xmin, ymax, zmin],
        ],
        [
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmax, ymin, zmin],
        ],
        [
            [xmin, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
            [xmax, ymin, zmax],
        ],
    ]:
        points = []
        for coord in coords:
            points.append(model.add_point(coord, mesh_size=resolution))
        channel_lines = [
            model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
        ]
        channel_loop = model.add_curve_loop(channel_lines)
        channel_surfaces.append(model.add_plane_surface(channel_loop))
    surface_loop = model.add_surface_loop(channel_surfaces)
    return channel_surfaces, surface_loop


def mesh3D(
    component: ComponentOrReference,
    base_resolution: float = 0.2,
    refine_resolution: Optional[Dict[Layer, float]] = None,
    padding: Tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ),
    layer_stack: Optional[LayerStack] = None,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
):
    """Returns gmsh 3D geometry of component.

    Similar to component.to_3d(), but returns a gmsh mesh with:
    - layer-dependent mesh resolution
    - different physical blocks for different objects
    - sub-object labels for introduction in physical solvers (e.g. edges)

    Args:
        component: Component or ComponentReference.
        base_resolution: background mesh resolution (um).
        refine_resolution: feature mesh resolution (um); layer dependent via a dict (default to base_resolution).
        padding: amount (west, east, south, north, down, up) to enlarge simulation region beyond features (um).

    """
    layer_stack = layer_stack or get_layer_stack()
    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    exclude_layers = exclude_layers or ()

    geometry = pygmsh.occ.geometry.Geometry()
    geometry.characteristic_length_min = base_resolution
    geometry.characteristic_length_max = base_resolution

    model = geometry.__enter__()

    zmin_cell = np.inf
    zmax_cell = -np.inf
    xmin_cell = np.inf
    ymin_cell = np.inf
    xmax_cell = -np.inf
    ymax_cell = -np.inf

    # Create element resolution dict
    refine_dict = {}
    for layer in component.get_layers():
        if layer in refine_resolution.keys():
            refine_dict[layer] = refine_resolution[layer]
        else:
            refine_dict[layer] = base_resolution

    # Features
    all_blocks = []
    for layer, polygons in component.get_polygons(by_spec=True).items():
        layer_blocks = []
        if (
            layer not in exclude_layers
            and layer in layer_to_thickness
            and layer in layer_to_zmin
        ):
            height = layer_to_thickness[layer]
            zmin_layer = layer_to_zmin[layer]
            zmax_layer = zmin_layer + height

            if zmin_layer < zmin_cell:
                zmin_cell = zmin_layer
            if zmax_layer > zmax_cell:
                zmax_cell = zmax_layer

            num_layers = int(height/refine_dict[layer])

            i = 0
            for polygon in polygons:
                xmin_block = np.min(polygon[:,0])
                xmax_block = np.max(polygon[:,0])
                ymin_block = np.min(polygon[:,1])
                ymax_block = np.max(polygon[:,1])
                zmin_block = zmin_layer
                zmax_block = zmax_layer
                if xmin_block < xmin_cell:
                    xmin_cell = xmin_block
                if xmax_block > xmax_cell:
                    xmax_cell = xmax_block
                if ymin_block < ymin_cell:
                    ymin_cell = ymin_block
                if ymax_block > ymax_cell:
                    ymax_cell = ymax_block
                layer_blocks.append(model.add_box([xmin_block, ymin_block, zmin_block], [xmax_block-xmin_block, ymax_block-ymin_block, zmax_block-zmin_block], mesh_size=refine_dict[layer]))
                i += 1

            # Recursively compute boolean fragments to eliminate overlaps
            for block_index, block in enumerate(layer_blocks):
                if block_index == 0:
                    continue
                else:
                    block = model.boolean_fragments(block, layer_blocks[block_index-1], delete_first=True, delete_other=True)

            model.add_physical(block, f"{layer}")
            all_blocks.append(block)

    xmin_cell -= padding[0]
    xmax_cell += padding[1]
    ymin_cell -= padding[2]
    ymax_cell += padding[3]
    zmin_cell -= padding[4]
    zmax_cell += padding[5]

    # Background oxide
    volume = model.add_box([xmin_cell, ymin_cell, zmin_cell], [xmax_cell-xmin_cell, ymax_cell-ymin_cell, zmax_cell-zmin_cell], mesh_size=base_resolution)
    for block in all_blocks:
            volume = model.boolean_fragments(volume, block, delete_first=False, delete_other=False)

    model.add_physical(volume, "oxide")
    geometry.generate_mesh(dim=3, verbose=True)

    return geometry


if __name__ == "__main__":

    import gdsfactory as gf

    # heaters = gf.Component("heaters")
    heater1 = gf.components.straight_heater_metal(length=2)
    # heater2 = gf.components.straight_heater_metal(length=2).move([0, 20])

    # heaters = gf.Component()
    # heaters << heater1
    # heaters << heater2
    heater1.show()

    layers_to_keep = [(1,0), (47,0)]

    geometry = mesh3D(
        heater1,
        exclude_layers=[layer_name for layer_name in heater1.get_layers() if layer_name not in layers_to_keep],
        refine_resolution={(1, 0): 0.05, (47, 0): 0.2},
        base_resolution=1,
        padding=[2,2,2,2,2,2]
    )

    import gmsh

    gmsh.write("mesh.msh")
    gmsh.clear()
    geometry.__exit__()

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

    # line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    # meshio.write("facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "tetra", prune_z=False)
    meshio.write("mesh.xdmf", triangle_mesh)

    # for layer, polygons in heaters.get_polygons(by_spec=True).items():
    #     print(layer, polygons)
