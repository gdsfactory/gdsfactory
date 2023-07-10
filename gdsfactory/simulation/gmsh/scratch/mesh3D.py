from __future__ import annotations

from typing import Dict, Optional, Tuple

import pygmsh

from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentOrReference, Layer


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
        points = [model.add_point(coord, mesh_size=resolution) for coord in coords]
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

    model = geometry.__enter__()

    zmin_cell = 0  # np.inf
    zmax_cell = 1.22  # -np.inf

    bbox = component.bbox
    xmin_cell = bbox[0][0] - padding[0]
    ymin_cell = bbox[0][1] - padding[2]
    xmax_cell = bbox[1][0] + padding[1]
    ymax_cell = bbox[1][1] + padding[3]

    # Create element resolution dict
    # refine_dict = {
    #     layer: refine_resolution[layer]
    #     if layer in refine_resolution.keys()
    #     else base_resolution
    #     for layer in component.get_layers()
    # }

    # Features
    # blocks = []
    # for layer, polygons in component.get_polygons(by_spec=True).items():
    for layer in component.layers:
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

            # num_layers = int(height/refine_dict[layer])
            # i = 0
            # for polygon in polygons:
            #     points = [model.add_point([polygon_point[0], polygon_point[1], zmin_layer], mesh_size=refine_dict[layer]) for polygon_point in polygon]
            #     polygon_lines = [
            #         model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
            #     ]
            #     polygon_loop = model.add_curve_loop(polygon_lines)
            #     polygon_surface = model.add_plane_surface(polygon_loop)
            #     polygon_top, polygon_volume, polygon_lat = model.extrude(polygon_surface, [0,0,height], num_layers=int(height/refine_dict[layer]))
            #     model.add_physical(polygon_volume, f"{layer}_{i}")
            #     blocks.append("{layer}_{i}")
            #     i += 1

    zmin_cell -= padding[4]
    zmax_cell += padding[5]

    # Background oxide
    # Generate boundary surfaces
    # cell_planes = []
    # for coords in [
    #                 [
    #                     [xmin_cell,ymin_cell,zmin_cell],
    #                     [xmin_cell,ymin_cell,zmax_cell],
    #                     [xmin_cell,ymax_cell,zmax_cell],
    #                     [xmin_cell,ymax_cell,zmin_cell]
    #                 ],
    #                 [
    #                     [xmax_cell,ymin_cell,zmin_cell],
    #                     [xmax_cell,ymin_cell,zmax_cell],
    #                     [xmax_cell,ymax_cell,zmax_cell],
    #                     [xmax_cell,ymax_cell,zmin_cell]
    #                 ],
    #                 [
    #                     [xmax_cell,ymin_cell,zmin_cell],
    #                     [xmax_cell,ymin_cell,zmax_cell],
    #                     [xmin_cell,ymin_cell,zmax_cell],
    #                     [xmin_cell,ymin_cell,zmin_cell]
    #                 ],
    #                 [
    #                     [xmax_cell,ymax_cell,zmin_cell],
    #                     [xmax_cell,ymax_cell,zmax_cell],
    #                     [xmin_cell,ymax_cell,zmax_cell],
    #                     [xmin_cell,ymax_cell,zmin_cell]
    #                 ],
    #                 [
    #                     [xmin_cell, ymin_cell, zmin_cell],
    #                     [xmin_cell, ymax_cell, zmin_cell],
    #                     [xmax_cell, ymax_cell, zmin_cell],
    #                     [xmax_cell, ymin_cell, zmin_cell]
    #                 ],
    #                 [
    #                     [xmin_cell, ymin_cell, zmax_cell],
    #                     [xmin_cell, ymax_cell, zmax_cell],
    #                     [xmax_cell, ymax_cell, zmax_cell],
    #                     [xmax_cell, ymin_cell, zmax_cell]
    #                 ],
    #             ]:
    #     points = []
    #     for coord in coords:
    #         points.append(model.add_point(coord, mesh_size=base_resolution))
    #     channel_lines = [
    #         model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
    #     ]
    #     channel_loop = model.add_curve_loop(channel_lines)
    #     plane_surface = model.add_plane_surface(channel_loop)
    #     cell_planes.append(plane_surface)

    channel_surfaces, surface_loop = surface_loop_from_vertices(
        model,
        xmin=xmin_cell,
        xmax=xmax_cell,
        ymin=ymin_cell,
        ymax=ymax_cell,
        zmin=0,
        zmax=1,
        resolution=0.1,
    )
    # i = 0

    cell_volume = model.add_volume(surface_loop)  # , holes=blocks)
    # for surface in channel_surfaces:
    #     geometry.boolean_fragments([cell_volume, surface], [], delete_first=True, delete_other=False)
    # model.add_physical(surface, f"oxide_{i}")
    # i += 1
    # cell_volume = model.boolean_difference(cell_volume, channel_surfaces, delete_first=True, delete_other=False)
    cell_volume = model.boolean_fragments(
        cell_volume, channel_surfaces, delete_first=True, delete_other=True
    )
    # model.add_physical(cell_volume, "oxide_vol")

    # for coords in [[
    #                 [xmin_cell, ymin_cell, zmin_cell],
    #                 [xmin_cell, ymax_cell, zmin_cell],
    #                 [xmax_cell, ymax_cell, zmin_cell],
    #                 [xmax_cell, ymin_cell, zmin_cell]
    #             ]]:
    #     points = []
    #     for coord in coords:
    #         points.append(model.add_point(coord, mesh_size=base_resolution))

    #     channel_lines = [
    #         model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
    #     ]
    #     channel_loop = model.add_curve_loop(channel_lines)
    #     plane_surface = model.add_plane_surface(channel_loop)

    # top, volume, lat = model.extrude(plane_surface, [0,0,zmax_cell-zmin_cell], num_layers=int((zmax_cell-zmin_cell)/base_resolution))
    # bottom = plane_surface

    # model.add_physical(cell_volume, "oxide")
    # model.add_physical(lat, "lat")
    # model.add_physical(top, "top")
    # model.add_physical(bottom, "bottom")
    geometry.generate_mesh(dim=3, verbose=True)

    return geometry


if __name__ == "__main__":
    import gdsfactory as gf

    heaters = gf.Component("heaters")
    heater1 = gf.components.straight(length=2)
    heater2 = gf.components.straight(length=2).move([0, 1])

    heaters = gf.Component()
    heaters << heater1
    heaters << heater2
    heaters.show()

    print(heaters.get_layers())

    geometry = mesh3D(
        heaters,
        exclude_layers=[(1, 10)],
        refine_resolution={(1, 0): 0.05, (47, 0): 0.05},
    )

    import gmsh

    gmsh.write("mesh.msh")
    gmsh.clear()
    geometry.__exit__()

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

    # line_mesh = create_mesh(mesh_from_file, "line")
    # meshio.write("facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle")
    meshio.write("mesh.xdmf", triangle_mesh)

    # for layer, polygons in heaters.get_polygons(by_spec=True).items():
    #     print(layer, polygons)
