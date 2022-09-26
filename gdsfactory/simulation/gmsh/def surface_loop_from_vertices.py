def surface_loop_from_vertices(
    model,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    resolution
):
    """Returns surface loop of prism from bounding box

    Args:
        

    """
    channel_surfaces = []
    for coords in [
                    [
                        [xmin,ymin,zmin],
                        [xmin,ymin,zmax],
                        [xmin,ymax,zmax],
                        [xmin,ymax,zmin]
                    ],
                    [
                        [xmax,ymin,zmin],
                        [xmax,ymin,zmax],
                        [xmax,ymax,zmax],
                        [xmax,ymax,zmin]
                    ],
                    [
                        [xmax,ymin,zmin],
                        [xmax,ymin,zmax],
                        [xmin,ymin,zmax],
                        [xmin,ymin,zmin]
                    ],
                    [
                        [xmax,ymax,zmin],
                        [xmax,ymax,zmax],
                        [xmin,ymax,zmax],
                        [xmin,ymax,zmin]
                    ],
                    [
                        [xmin, ymin, zmin],
                        [xmin, ymax, zmin],
                        [xmax, ymax, zmin],
                        [xmax, ymin, zmin]
                    ],
                    [
                        [xmin, ymin, zmax],
                        [xmin, ymax, zmax],
                        [xmax, ymax, zmax],
                        [xmax, ymin, zmax]
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
    padding: Tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    layer_stack: Optional[LayerStack] = None,
    exclude_layers: Optional[Tuple[Layer, ...]] = None,
):
    """Returns gmsh 3D geometry of component

    Similar to component.to_3d(), but returns a **volumetric** gmsh mesh with: 
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

    geometry = pygmsh.geo.Geometry()

    model = geometry.__enter__()

    zmin_cell = np.inf
    zmax_cell = -np.inf

    bbox = component.bbox
    xmin_cell = bbox[0][0] - padding[0]
    ymin_cell = bbox[0][1] - padding[2]
    xmax_cell = bbox[1][0] + padding[2]
    ymax_cell = bbox[1][1] + padding[3]

    # Create element resolution dict
    refine_dict = {}
    for layer in component.get_layers():
        if layer in refine_resolution.keys():
            refine_dict[layer] = refine_resolution[layer]
        else:
            refine_dict[layer] = base_resolution

    # Features
    blocks = []
    for layer, polygons in component.get_polygons(by_spec=True).items():
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
                points = [model.add_point([polygon_point[0], polygon_point[1], zmin_layer], mesh_size=refine_dict[layer]) for polygon_point in polygon]
                polygon_lines = [
                    model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
                ]
                polygon_loop = model.add_curve_loop(polygon_lines)
                polygon_surface = model.add_plane_surface(polygon_loop)
                polygon_top, polygon_volume, polygon_lat = model.extrude(polygon_surface, [0,0,height], num_layers=int(height/refine_dict[layer]))
                model.add_physical(polygon_volume, f"{layer}_{i}")
                # Generate surface loops
                blocks.append(polygon_volume)
                i += 1

    zmin_cell -= padding[4]
    zmax_cell += padding[5]

    # Background oxide
    # Generate boundary surfaces
    cell_surfaces, cell_surfaceloop = model.add_surface_loop(surface_loop_from_vertices(
        model=model,
        xmin=xmin_cell,
        xmax=xmax_cell,
        ymin=ymin_cell,
        ymax=ymax_cell,
        zmin=zmin_cell,
        zmax=zmax_cell,
        resolution=base_resolution
        ))
    cell_volume = model.add_volume(cell_surfaces)#, holes=blocks)