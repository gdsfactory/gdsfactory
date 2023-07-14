from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from gdsfactory.simulation.gmsh.parse_component import bufferize
from gdsfactory.simulation.gmsh.parse_gds import cleanup_component
from gdsfactory.simulation.gmsh.parse_layerstack import order_layerstack
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentOrReference

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
    global_scaling: float = 1,
    filename: Optional[str] = None,
    verbosity: Optional[int] = 0,
    round_tol: int = 3,
    simplify_tol: float = 1e-3,
) -> bool:
    """Full 3D mesh of component.

    Args:
        component (Component): gdsfactory component to mesh
        layerstack (LayerStack): gdsfactory LayerStack to parse
        resolutions (Dict): Pairs {"layername": {"resolution": float, "distance": "float}} to roughly control mesh refinement
        default_characteristic_length (float): gmsh maximum edge length
        global_scaling: factor to scale all mesh coordinates by (e.g. 1E-6 to go from um to m)
        filename (str, path): where to save the .msh file
        round_tol: during gds --> mesh conversion cleanup, number of decimal points at which to round the gdsfactory/shapely points before introducing to gmsh
        simplify_tol: during gds --> mesh conversion cleanup, shapely "simplify" tolerance (make it so all points are at least separated by this amount)

    """
    # Fuse and cleanup polygons of same layer in case user overlapped them
    layer_polygons_dict = cleanup_component(
        component, layerstack, round_tol, simplify_tol
    )

    # Meshwell Prisms from gdsfactory polygons and layerstack
    model = Model()
    prisms_dict = define_prisms(layer_polygons_dict, layerstack, model)

    # Mesh
    mesh_out = model.mesh(
        entities_dict=prisms_dict,
        resolutions=resolutions,
        default_characteristic_length=default_characteristic_length,
        global_scaling=global_scaling,
        filename=filename,
        verbosity=verbosity,
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
    c = layerstack.get_component_with_net_layers(
        c,
        portnames=["r_e2", "l_e4"],
        delimiter="#",
    )

    # FIXME: .filtered returns all layers
    # filtered_layerstack = layerstack.filtered_from_layerspec(layerspecs=c.get_layers())
    filtered_layerstack = LayerStack(
        layers={
            k: layerstack.layers[k]
            for k in (
                "via1",
                "clad",
                "metal2",
                "metal3#l_e4",
                "heater",
                "via2",
                "core",
                "metal3#r_e2",
            )
        }
    )

    resolutions = {
        "core": {"resolution": 0.1},
    }
    geometry = xyz_mesh(
        component=c,
        layerstack=filtered_layerstack,
        resolutions=resolutions,
        filename="mesh.msh",
        verbosity=5,
    )
