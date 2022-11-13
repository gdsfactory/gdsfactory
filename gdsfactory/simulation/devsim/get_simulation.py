from typing import Optional, Tuple

import numpy as np
from devsim import (
    add_gmsh_region,
    create_device,
    create_gmsh_mesh,
    finalize_mesh,
    get_node_model_values,
    node_solution,
    set_node_values,
    write_devices,
)

from gdsfactory import Component
from gdsfactory.simulation.devsim.doping import get_doping_info_generic
from gdsfactory.simulation.gmsh import (
    fuse_component_layer,
    get_u_bounds_polygons,
    uz_xsection_mesh,
)
from gdsfactory.tech import LayerStack


def create_2Duz_simulation(
    component: Component,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    physical_layerstack: LayerStack,
    doping_info,  # Dict[str, DopingLayerLevel],
    background_tag: Optional[str] = None,
    temp_file_name="temp.msh2",
    devsim_mesh_name="temp",
    devsim_device_name="temp",
    devsim_mesh_file_name="devsim.dat",
):
    # Get structural mesh
    uz_xsection_mesh(
        component,
        xsection_bounds,
        physical_layerstack,
        resolutions=resolutions,
        background_tag=background_tag,
        filename=temp_file_name,
    )

    # Get doping layer bounds
    doping_polygons = {}
    for layername, layer_obj in doping_info.items():
        fused_polygons = fuse_component_layer(component, layer_obj.layer)
        bounds = get_u_bounds_polygons(fused_polygons, xsection_bounds)
        doping_polygons[layername] = bounds

    # Define structural mesh in DEVSIM
    create_gmsh_mesh(file=temp_file_name, mesh=devsim_mesh_name)
    physical_layerstack_dict = physical_layerstack.to_dict()
    for name, values in physical_layerstack_dict.items():
        add_gmsh_region(
            mesh=devsim_mesh_name,
            gmsh_name=name,
            region=values["material"],
            material=values["material"],
        )
    if background_tag:
        add_gmsh_region(
            mesh=devsim_mesh_name,
            gmsh_name=background_tag,
            region=background_tag,
            material=background_tag,
        )
    finalize_mesh(mesh=devsim_mesh_name)
    create_device(mesh=devsim_mesh_name, device=devsim_device_name)

    # Assign doping fields to the structural mesh
    # Hardcoded silicon only now
    xpos = get_node_model_values(device=devsim_device_name, region="si", name="x")
    # ypos = get_node_model_values(device=devsim_device_name, region="si", name="y")

    acceptor = np.zeros_like(xpos)
    donor = np.zeros_like(xpos)
    for layername, bounds in doping_polygons.items():
        for bound in bounds:
            node_inds = np.intersect1d(
                np.where(xpos >= bound[0] * np.ones_like(xpos)),
                np.where(xpos <= bound[1] * np.ones_like(xpos)),
            )
            # Assume step doping for now (does not depend on y, or z)
            if doping_info[layername].type == "Acceptor":
                acceptor[node_inds] += doping_info[layername].z_profile(0)
            elif doping_info[layername].type == "Donor":
                donor[node_inds] += doping_info[layername].z_profile(0)
            else:
                raise ValueError(
                    f'Doping type "{doping_info[layername].type}" not supported.'
                )

    net_doping = donor - acceptor

    node_solution(device=devsim_device_name, region="si", name="Acceptors")
    set_node_values(
        device=devsim_device_name, region="si", name="Acceptors", values=acceptor
    )
    node_solution(device=devsim_device_name, region="si", name="Donors")
    set_node_values(device=devsim_device_name, region="si", name="Donors", values=donor)
    node_solution(device=devsim_device_name, region="si", name="Acceptors")
    node_solution(device=devsim_device_name, region="si", name="NetDoping")
    set_node_values(
        device=devsim_device_name, region="si", name="NetDoping", values=net_doping
    )

    write_devices(file=devsim_mesh_file_name, type="tecplot")

    return


if __name__ == "__main__":

    import gdsfactory as gf
    from gdsfactory.tech import LayerStack, get_layer_stack_generic

    waveguide = gf.components.straight_pn(length=10, taper=None)
    waveguide = gf.geometry.trim(
        component=waveguide, domain=[[3, -4], [3, 4], [5, 4], [5, -4]]
    )
    # We add simulation layers for contacts
    waveguide.show()

    physical_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
                "via_contact",
                # "metal2",
            )  # "slab90", "via_contact")#"via_contact") # "slab90", "core"
        }
    )

    resolutions = {}
    resolutions["core"] = {"resolution": 0.01, "distance": 2}
    resolutions["slab90"] = {"resolution": 0.03, "distance": 1}
    resolutions["via_contact"] = {"resolution": 0.1, "distance": 1}

    physical_mesh = uz_xsection_mesh(
        waveguide,
        [(4, -4), (4, 4)],
        physical_layerstack,
        resolutions=resolutions,
        background_tag="Oxide",
        filename="temp.msh2",
    )

    create_2Duz_simulation(
        component=waveguide,
        xsection_bounds=[(4, -4), (4, 4)],
        physical_layerstack=physical_layerstack,
        doping_info=get_doping_info_generic(),
        background_tag="Oxide",
    )

    # We also need the doping layer locations for the component
    # layermap = gf.tech.LayerMap()
    # doping_layers = {"N": layermap.N, "P": layermap.P, "NNN": layermap.NPP, "PPP": layermap.PPP}
    # doping_polygons = {}
    # for doping_layername, doping_layer in doping_layers.items():
    #     doping_polygons[doping_layername] = waveguide.extract(doping_layer).get_polygons()

    # for doping_layername, doping_polygons in doping_polygons.items():
    #     print(doping_layername, doping_polygons)

    # print(layermap)
