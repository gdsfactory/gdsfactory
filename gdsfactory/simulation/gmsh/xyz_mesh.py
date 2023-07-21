from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from gdsfactory.config import get_number_of_cores
from gdsfactory.simulation.gmsh.parse_component import bufferize
from gdsfactory.simulation.gmsh.parse_gds import cleanup_component
from gdsfactory.simulation.gmsh.parse_layerstack import (
    order_layerstack,
    list_unique_layerstack_z,
)
from gdsfactory.technology import LayerStack, LayerLevel
from gdsfactory.typings import ComponentOrReference, List
from gdsfactory.geometry.union import union

from meshwell.prism import Prism
from meshwell.model import Model

from collections import OrderedDict


def define_prisms(layer_polygons_dict, layerstack, model):
    """Define meshwell prism dimtags from gdsfactory information."""
    prisms_dict = OrderedDict()
    buffered_layerstack = bufferize(layerstack)
    ordered_layerstack = order_layerstack(layerstack)

    for layername in ordered_layerstack:
        coords = np.array(buffered_layerstack.layers[layername].z_to_bias[0])
        zs = (
            coords * buffered_layerstack.layers[layername].thickness
            + buffered_layerstack.layers[layername].zmin
        )
        buffers = buffered_layerstack.layers[layername].z_to_bias[1]

        buffer_dict = dict(zip(zs, buffers))

        prisms_dict[layername] = Prism(
            polygons=layer_polygons_dict[layername],
            buffers=buffer_dict,
            model=model,
        )

    return prisms_dict


def xyz_mesh(
    component: ComponentOrReference,
    layerstack: LayerStack,
    resolutions: Optional[Dict] = None,
    default_characteristic_length: float = 0.5,
    background_tag: Optional[str] = None,
    background_padding: Sequence[float, float, float, float, float, float] = (2.0,) * 6,
    global_scaling: float = 1,
    global_2D_algorithm: int = 6,
    global_3D_algorithm: int = 1,
    filename: Optional[str] = None,
    verbosity: Optional[int] = 0,
    round_tol: int = 3,
    simplify_tol: float = 1e-3,
    n_threads: int = get_number_of_cores(),
    portnames: List[str] = None,
    layer_portname_delimiter: str = "#",
) -> bool:
    """Full 3D mesh of component.

    Args:
        component: gdsfactory component to mesh
        layerstack: gdsfactory LayerStack to parse
        resolutions: Pairs {"layername": {"resolution": float, "distance": "float}} to roughly control mesh refinement
        default_characteristic_length: gmsh maximum edge length
        background_tag: name of the background layer to add (default: no background added). This will be used as the material as well.
        background_padding: [-x, -y, -z, +x, +y, +z] distances to add to the components and to fill with ``background_tag``
        global_scaling: factor to scale all mesh coordinates by (e.g. 1E-6 to go from um to m)
        global_2D_algorithm: gmsh surface default meshing algorithm, see https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
        global_3D_algorithm: gmsh volume default meshing algorithm, see https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
        filename: where to save the .msh file
        round_tol: during gds --> mesh conversion cleanup, number of decimal points at which to round the gdsfactory/shapely points before introducing to gmsh
        simplify_tol: during gds --> mesh conversion cleanup, shapely "simplify" tolerance (make it so all points are at least separated by this amount)
        n_threads: for gmsh parallelization
        portnames: list or port polygons to converts into new layers (useful for boundary conditions)
        layer_portname_delimiter: delimiter for the new layername/portname physicals, formatted as {layername}{delimiter}{portname}
    """
    if portnames:
        mesh_component = gf.Component()
        mesh_component << union(component, by_layer=True)
        mesh_component.add_ports(component.get_ports_list())
        component = layerstack.get_component_with_net_layers(
            mesh_component,
            portnames=portnames,
            delimiter=layer_portname_delimiter,
        )

    # Fuse and cleanup polygons of same layer in case user overlapped them
    # TODO: some duplication with union above, although this also does some useful offsetting
    layer_polygons_dict = cleanup_component(
        component, layerstack, round_tol, simplify_tol
    )

    # Add background polygon
    if background_tag is not None:
        bbox = unary_union(list(layer_polygons_dict.values()))
        bounds = bbox.bounds

        # get min and max z values in LayerStack
        zs = list_unique_layerstack_z(layerstack)
        zmin, zmax = np.min(zs), np.max(zs)

        # create Polygon encompassing simulation environment
        layer_polygons_dict[background_tag] = Polygon(
            [
                [bounds[0] - background_padding[0], bounds[1] - background_padding[1]],
                [bounds[0] - background_padding[0], bounds[3] + background_padding[4]],
                [bounds[2] + background_padding[3], bounds[3] + background_padding[4]],
                [bounds[2] + background_padding[3], bounds[1] - background_padding[1]],
            ]
        )
        layerstack = LayerStack(
            layers=layerstack.layers
            | {
                background_tag: LayerLevel(
                    layer=(9999, 0),  # TODO something like LAYERS.BACKGROUND?
                    thickness=(zmax + background_padding[5])
                    - (zmin - background_padding[2]),
                    zmin=zmin - background_padding[2],
                    material=background_tag,
                    mesh_order=2**63 - 1,
                )
            }
        )

    # Meshwell Prisms from gdsfactory polygons and layerstack
    model = Model(n_threads=n_threads)
    prisms_dict = define_prisms(layer_polygons_dict, layerstack, model)

    # Mesh
    mesh_out = model.mesh(
        entities_dict=prisms_dict,
        resolutions=resolutions,
        default_characteristic_length=default_characteristic_length,
        global_scaling=global_scaling,
        global_2D_algorithm=global_2D_algorithm,
        global_3D_algorithm=global_3D_algorithm,
        filename=filename,
        verbosity=verbosity,
        n_threads=n_threads,
    )

    return mesh_out


if __name__ == "__main__":
    import gdsfactory as gf

    from gdsfactory.pdk import get_layer_stack
    from gdsfactory.generic_tech import LAYER

    # Choose some component
    c = gf.component.Component()
    waveguide = c << gf.get_component(gf.components.straight_heater_metal(length=40))
    c.add_ports(waveguide.get_ports_list())

    # Add wafer / vacuum (could be automated)
    wafer = c << gf.components.bbox(bbox=waveguide.bbox, layer=LAYER.WAFER)

    # Generate a new component and layerstack with new logical layers
    layerstack = get_layer_stack()

    # FIXME: .filtered returns all layers
    # filtered_layerstack = layerstack.filtered_from_layerspec(layerspecs=c.get_layers())
    filtered_layerstack = LayerStack(
        layers={
            k: layerstack.layers[k]
            for k in (
                # "via1",
                "box",
                "clad",
                # "metal2",
                "heater",
                "via2",
                "core",
                "metal3",
                # "via_contact",
                # "metal1"
            )
        }
    )

    resolutions = {
        "core": {"resolution": 0.3},
    }
    geometry = xyz_mesh(
        component=c,
        layerstack=filtered_layerstack,
        resolutions=resolutions,
        filename="mesh.msh",
        default_characteristic_length=5,
        verbosity=5,
        portnames=["r_e2", "l_e4"],
    )
