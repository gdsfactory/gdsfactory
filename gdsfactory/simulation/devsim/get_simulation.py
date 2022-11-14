from typing import Dict, Optional, Tuple

from devsim import (
    add_gmsh_region,
    create_device,
    create_gmsh_mesh,
    finalize_mesh,
    write_devices,
)

from gdsfactory import Component
from gdsfactory.simulation.devsim.doping import get_doping_info_generic
from gdsfactory.simulation.gmsh import (
    fuse_component_layer,
    get_u_bounds_polygons,
    uz_xsection_mesh,
)
from gdsfactory.tech import LayerLevel, LayerStack


def create_2Duz_simulation(
    component: Component,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    physical_layerstack: LayerStack,
    doping_info,  # Dict[str, DopingLayerLevel],
    contact_info,
    resolutions: Optional[Dict[str, Dict]] = {},
    background_tag: Optional[str] = None,
    temp_file_name="temp.msh2",
    devsim_mesh_name="temp",
    devsim_device_name="temp",
    devsim_mesh_file_name="devsim.dat",
):
    # Add contacts to simulation
    simulation_layertack = physical_layerstack
    for contact_name, contact_dict in contact_info.items():
        layerlevel = LayerLevel(
            layer=contact_dict["gds_layer"],
            thickness=contact_dict["physical_layerlevel"].thickness,
            zmin=contact_dict["physical_layerlevel"].zmin,
            material=contact_dict["physical_layerlevel"].material,
            sidewall_angle=contact_dict["physical_layerlevel"].sidewall_angle,
            info=contact_dict["physical_layerlevel"].info,
        )
        simulation_layertack.layers[contact_name] = layerlevel

    # Get structural mesh
    uz_xsection_mesh(
        component,
        xsection_bounds,
        simulation_layertack,
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

    # Define physical structural mesh in DEVSIM
    create_gmsh_mesh(file=temp_file_name, mesh=devsim_mesh_name)
    simulation_layerstack_dict = simulation_layertack.to_dict()
    for name, values in simulation_layerstack_dict.items():
        add_gmsh_region(
            mesh=devsim_mesh_name,
            gmsh_name=name,
            region=name,
            material=values["material"],
        )
    if background_tag:
        add_gmsh_region(
            mesh=devsim_mesh_name,
            gmsh_name=background_tag,
            region=background_tag,
            material=background_tag,
        )
    # Contacts
    # for contact_name, contact in contact_info.items():
    #     add_gmsh_contact(
    #         gmsh_name=contact_name,
    #         material=simulation_layertack.layers[contact_name].material,
    #         mesh=devsim_mesh_name,
    #         name=contact_name,
    #         region=simulation_layertack.layers[contact_name].material,
    #     )

    finalize_mesh(mesh=devsim_mesh_name)
    create_device(mesh=devsim_mesh_name, device=devsim_device_name)

    # Assign doping fields to the structural mesh
    # for region_name in []:
    #     xpos = get_node_model_values(device=devsim_device_name, region="si", name="x")
    #     # ypos = get_node_model_values(device=devsim_device_name, region="si", name="y")

    #     acceptor = np.zeros_like(xpos)
    #     donor = np.zeros_like(xpos)
    #     for layername, bounds in doping_polygons.items():
    #         for bound in bounds:
    #             node_inds = np.intersect1d(
    #                 np.where(xpos >= bound[0] * np.ones_like(xpos)),
    #                 np.where(xpos <= bound[1] * np.ones_like(xpos)),
    #             )
    #             # Assume step doping for now (does not depend on y, or z)
    #             if doping_info[layername].type == "Acceptor":
    #                 acceptor[node_inds] += doping_info[layername].z_profile(0)
    #             elif doping_info[layername].type == "Donor":
    #                 donor[node_inds] += doping_info[layername].z_profile(0)
    #             else:
    #                 raise ValueError(
    #                     f'Doping type "{doping_info[layername].type}" not supported.'
    #                 )

    #     net_doping = donor - acceptor

    #     node_solution(device=devsim_device_name, region="si", name="Acceptors")
    #     set_node_values(
    #         device=devsim_device_name, region="si", name="Acceptors", values=acceptor
    #     )
    #     node_solution(device=devsim_device_name, region="si", name="Donors")
    #     set_node_values(device=devsim_device_name, region="si", name="Donors", values=donor)
    #     node_solution(device=devsim_device_name, region="si", name="Acceptors")
    #     node_solution(device=devsim_device_name, region="si", name="NetDoping")
    #     set_node_values(
    #         device=devsim_device_name, region="si", name="NetDoping", values=net_doping
    #     )

    # Define contacts

    write_devices(file=devsim_mesh_file_name, type="tecplot")

    return devsim_device_name, devsim_mesh_file_name


if __name__ == "__main__":

    import gdsfactory as gf
    from gdsfactory.tech import LayerStack, get_layer_stack_generic

    # We choose a representative subdomain of the component
    waveguide = gf.Component()

    waveguide.add_ref(
        gf.geometry.trim(
            component=gf.components.straight_pn(length=10, taper=None),
            domain=[[3, -4], [3, 4], [5, 4], [5, -4]],
        )
    )

    # We will restrict the physical to a subset of layers:
    layermap = gf.tech.LayerMap()
    physical_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
                # "metal2",
            )  # "slab90", "via_contact")#"via_contact") # "slab90", "core"
        }
    )

    # Boundary conditions such as contact are also defined with GDS polygons.
    # While complete dummy layers + stack definitions could be used, often we are interested in using existing layers as contact (for instance vias).
    # Since this means different contacts could be on the same physical layer, this requires more care in the meshing.

    # Simulation layers for contacts
    anode_layer = (105, 0)
    cathode_layer = (106, 0)

    # We will place the contacts where the vias intersect heavy doping
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

    contact_info = {}
    contact_info["anode"] = {
        "gds_layer": anode_layer,
        "physical_layerlevel": get_layer_stack_generic().layers["via_contact"],
    }
    contact_info["cathode"] = {
        "gds_layer": cathode_layer,
        "physical_layerlevel": get_layer_stack_generic().layers["via_contact"],
    }

    waveguide.show()

    resolutions = {}
    resolutions["core"] = {"resolution": 0.01, "distance": 2}
    resolutions["slab90"] = {"resolution": 0.03, "distance": 1}
    resolutions["via_contact"] = {"resolution": 0.1, "distance": 1}

    # physical_mesh = uz_xsection_mesh(
    #     waveguide,
    #     [(4, -4), (4, 4)],
    #     physical_layerstack,
    #     resolutions=resolutions,
    #     background_tag="Oxide",
    #     filename="temp.msh2",
    # )

    create_2Duz_simulation(
        component=waveguide,
        xsection_bounds=[(4, -4), (4, 4)],
        physical_layerstack=physical_layerstack,
        doping_info=get_doping_info_generic(),
        contact_info=contact_info,
        resolutions=resolutions,
        background_tag="Oxide",
    )
