from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
from devsim import (
    add_gmsh_contact,
    add_gmsh_interface,
    add_gmsh_region,
    create_device,
    create_gmsh_mesh,
    finalize_mesh,
    get_node_model_values,
    node_solution,
    set_node_values,
    write_devices,
)
from scipy.interpolate import NearestNDInterpolator

from gdsfactory import Component
from gdsfactory.simulation.devsim.doping import get_doping_info_generic
from gdsfactory.simulation.gmsh import (
    fuse_polygons,
    get_u_bounds_polygons,
    uz_xsection_mesh,
)
from gdsfactory.technology import LayerLevel, LayerStack

um_to_cm = 1e-4


def create_2Duz_simulation(
    component: Component,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    full_layerstack: LayerStack,
    physical_layerstack: LayerStack,
    doping_info,  # Dict[str, DopingLayerLevel],
    contact_info,
    resolutions: Optional[Dict[str, Dict]] = None,
    mesh_scaling_factor: float = um_to_cm,
    default_resolution_min: float = 0.001,
    default_resolution_max: float = 0.2,
    background_tag: Optional[str] = None,
    temp_file_name: str = "temp.msh2",
    devsim_mesh_name: str = "temp",
    devsim_device_name: str = "temp",
    devsim_simulation_filename: str = "devsim.dat",
    global_meshsize_array: Optional[np.array] = None,
    global_meshsize_interpolant_func: Optional[callable] = NearestNDInterpolator,
):
    # Replace relevant physical entities by contacts
    simulation_layertack = physical_layerstack
    for contact_name, contact_dict in contact_info.items():
        contact_layer = full_layerstack.layers[
            contact_dict["physical_layerlevel_to_replace"]
        ]
        layerlevel = LayerLevel(
            layer=contact_dict["gds_layer"],
            thickness=contact_layer.thickness,
            zmin=contact_layer.zmin,
            material=contact_layer.material,
            sidewall_angle=contact_layer.sidewall_angle,
            info=contact_layer.info,
        )
        simulation_layertack.layers[contact_name] = layerlevel

    # Get structural mesh
    mesh = uz_xsection_mesh(
        component,
        xsection_bounds,
        simulation_layertack,
        resolutions=resolutions,
        mesh_scaling_factor=mesh_scaling_factor,
        default_resolution_min=default_resolution_min,
        default_resolution_max=default_resolution_max,
        background_tag=background_tag,
        filename=temp_file_name,
        global_meshsize_array=global_meshsize_array,
        global_meshsize_interpolant_func=global_meshsize_interpolant_func,
    )

    # Get doping layer bounds
    doping_polygons = {}
    for layername, layer_obj in doping_info.items():
        fused_polygons = fuse_polygons(component, layername, layer_obj.layer)
        bounds = get_u_bounds_polygons(fused_polygons, xsection_bounds)
        doping_polygons[layername] = [np.array(bound) * um_to_cm for bound in bounds]

    # Regions, tagged by material
    regions = {}
    create_gmsh_mesh(file=temp_file_name, mesh=devsim_mesh_name)
    simulation_layerstack_dict = simulation_layertack.to_dict()
    for name, values in simulation_layerstack_dict.items():
        add_gmsh_region(
            mesh=devsim_mesh_name,
            gmsh_name=name,
            region=name,
            material=values["material"],
        )
        if values["material"] in regions:
            regions[values["material"]].append(name)
        else:
            regions[values["material"]] = [name]
    if background_tag:
        simulation_layerstack_dict[background_tag] = {"material": background_tag}
        add_gmsh_region(
            mesh=devsim_mesh_name,
            gmsh_name=background_tag,
            region=background_tag,
            material=background_tag,
        )
        regions[background_tag] = [background_tag]

    # Contacts
    contacts = {}
    for contact_name, contact in contact_info.items():
        layer1 = contact_name
        layer2 = contact["physical_layerlevel_to_contact"]
        interface = f"{layer1}___{layer2}"
        if interface not in mesh.cell_sets_dict.keys():
            interface = f"{layer2}___{layer1}"
        add_gmsh_contact(
            gmsh_name=interface,
            material=simulation_layertack.layers[contact_name].material,
            mesh=devsim_mesh_name,
            name=contact_name,
            region=layer2,
        )
        contacts[contact_name] = interface

    # Interfaces (that are not contacts), labeled by material-material
    interfaces = {}
    for (name1, values1), (name2, values2) in combinations(
        simulation_layerstack_dict.items(), 2
    ):
        interface = f"{name1}___{name2}"
        if interface not in mesh.cell_sets_dict.keys():
            interface = f"{name2}___{name1}"
            if interface not in mesh.cell_sets_dict.keys():
                continue
        # if interface not in contacts.values():
        add_gmsh_interface(
            gmsh_name=interface,
            mesh=devsim_mesh_name,
            name=interface,
            region0=name1,
            region1=name2,
        )
        if interface in interfaces:
            interfaces[(values1["material"], values2["material"])].append(interface)
        else:
            interfaces[(values1["material"], values2["material"])] = [interface]
    finalize_mesh(mesh=devsim_mesh_name)
    create_device(mesh=devsim_mesh_name, device=devsim_device_name)

    # Assign doping fields silicon
    for region_name in regions["si"]:
        xpos = get_node_model_values(
            device=devsim_device_name, region=region_name, name="x"
        )
        # ypos = get_node_model_values(device=devsim_device_name, region="si", name="y")
        acceptor = np.zeros_like(xpos)
        donor = np.zeros_like(xpos)
        for layername, bounds in doping_polygons.items():
            print(layername, bounds)
            for bound in bounds:
                print(bound)
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

        node_solution(device=devsim_device_name, region=region_name, name="Acceptors")
        set_node_values(
            device=devsim_device_name,
            region=region_name,
            name="Acceptors",
            values=acceptor,
        )
        node_solution(device=devsim_device_name, region=region_name, name="Donors")
        set_node_values(
            device=devsim_device_name, region=region_name, name="Donors", values=donor
        )
        node_solution(device=devsim_device_name, region=region_name, name="NetDoping")
        set_node_values(
            device=devsim_device_name,
            region=region_name,
            name="NetDoping",
            values=net_doping,
        )

    # Assign doping field contact
    for contact_name in contacts:
        xpos = get_node_model_values(
            device=devsim_device_name, region=contact_name, name="x"
        )
        node_solution(device=devsim_device_name, region=contact_name, name="NetDoping")
        set_node_values(
            device=devsim_device_name,
            region=contact_name,
            name="NetDoping",
            values=[0] * len(xpos),
        )

    # Cast to cm
    # for region_name in get_region_list(device=devsim_device_name):
    #     xpos = get_node_model_values(
    #         device=devsim_device_name, region=region_name, name="x"
    #     )
    #     set_node_values(
    #         device=devsim_device_name, region=region_name, name="x", values=np.array(xpos)*um_to_cm
    #     )
    #     ypos = get_node_model_values(
    #         device=devsim_device_name, region=region_name, name="y"
    #     )
    #     set_node_values(
    #         device=devsim_device_name, region=region_name, name="y", values=np.array(ypos)*um_to_cm
    #     )

    if devsim_simulation_filename:
        write_devices(file=devsim_simulation_filename, type="tecplot")

    return devsim_device_name, regions, interfaces


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.generic_tech import get_layer_stack_generic

    # We choose a representative subdomain of the component
    waveguide = gf.Component()
    waveguide.add_ref(
        gf.geometry.trim(
            component=gf.components.straight_pn(length=10, taper=None),
            domain=[[3, -4], [3, 4], [5, 4], [5, -4]],
        )
    )

    waveguide.show()

    # We will restrict the physical mesh to a subset of layers:
    layermap = gf.generic_tech.LayerMap()
    physical_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
            )
        }
    )

    # Boundary conditions such as contacts are also defined with GDS polygons.
    # In DEVSIM, contacts must be N-1D, where N is the simulation dimensionality.
    # While complete dummy layers + stack definitions could be used, often we are interested in using existing layers as contact (for instance vias).
    # Since this means different contacts could correspond to the same physical layers, this requires more care in the meshing.

    # Simulation layers for contacts
    anode_layer = (105, 0)
    cathode_layer = (106, 0)

    # We will generate contacts where the vias intersect heavy doping
    via_contact_locations = waveguide.extract(layermap.VIAC)
    NPP_location = waveguide.extract(layermap.NPP)
    PPP_location = waveguide.extract(layermap.PPP)
    waveguide.add_ref(
        gf.geometry.boolean(
            via_contact_locations, NPP_location, operation="AND", layer=anode_layer
        )
    )
    waveguide.add_ref(
        gf.geometry.boolean(
            via_contact_locations, PPP_location, operation="AND", layer=cathode_layer
        )
    )

    # Contact info contains (could be improved)
    # gds_layer to use for this contact
    # physical_layerlevel_1: physical to replace by the contact
    # physical_layerlevel_2: interface to physical_layerlevel_1 to define as the contact
    # two layers defining an interface, and the dummy layer for position refinement
    contact_info = {
        "anode": {
            "gds_layer": anode_layer,
            "physical_layerlevel_to_replace": "via_contact",
            "physical_layerlevel_to_contact": "slab90",
        },
        "cathode": {
            "gds_layer": cathode_layer,
            "physical_layerlevel_to_replace": "via_contact",
            "physical_layerlevel_to_contact": "slab90",
        },
    }

    waveguide.show()

    resolutions = {
        "core": {"resolution": 0.01, "distance": 2},
        "slab90": {"resolution": 0.03, "distance": 1},
    }

    # resolutions["via_contact"] = {"resolution": 0.1, "distance": 1}

    device_name, regions, interfaces = create_2Duz_simulation(
        component=waveguide,
        xsection_bounds=[(4, -4), (4, 4)],
        full_layerstack=get_layer_stack_generic(),
        physical_layerstack=physical_layerstack,
        doping_info=get_doping_info_generic(),
        contact_info=contact_info,
        resolutions=resolutions,
        mesh_scaling_factor=1e-4,
        background_tag="sio2",
    )

    print(device_name)
    print(regions)
    print(interfaces)
