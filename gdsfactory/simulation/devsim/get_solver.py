from typing import Dict, Optional, Tuple

import numpy as np
from devsim import (
    get_contact_list,
    get_interface_list,
    get_region_list,
    set_node_values,
    set_parameter,
    solve,
)
from devsim.python_packages import model_create, simple_physics
from pydantic import BaseModel, Extra

from gdsfactory import Component
from gdsfactory.simulation.devsim import create_2Duz_simulation
from gdsfactory.simulation.devsim.doping import get_doping_info_generic
from gdsfactory.tech import LayerStack


def SetUniversalParameters(device, region):
    universal = {
        "q": 1.6e-19,  # , 'coul'),
        "k": 1.3806503e-23,  # , 'J/K'),
        "Permittivity_0": 8.85e-14,  # , 'F/cm^2')
    }
    for k, v in universal.items():
        set_parameter(device=device, region=region, name=k, value=v)


class DDComponent(BaseModel):
    """Test."""

    component: Component
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    full_layerstack: LayerStack
    physical_layerstack: LayerStack
    resolutions: Optional[Dict[str, Dict]] = {}
    background_tag: Optional[str] = None
    temp_file_name = "temp.msh2"
    devsim_mesh_name = "temp"
    devsim_device_name = "temp"
    devsim_simulation_filename = "devsim.dat"
    atol: float = 1e8
    rtol: float = 1e-8
    max_iter: int = 60

    class Config:
        """Enable adding new."""

        extra = Extra.allow

    def set_parameters(self, device, regions) -> None:

        for material, region_names in regions.items():
            for region_name in region_names:
                SetUniversalParameters(device, region_name)
                if material == "si":
                    simple_physics.SetSiliconParameters(device, region_name, T=300)
                elif material == "sio2":
                    continue
                    # physics.SetOxideParameters(device, region_name)
                elif material == "Aluminum":
                    # Change to aluminum eventually
                    simple_physics.SetSiliconParameters(device, region_name, T=300)

    def initial_solution(self, device, region, circuit_contacts=None) -> None:
        # Create Potential, Potential@n0, Potential@n1
        model_create.CreateSolution(device, region, "Potential")

        # Create potential only physical models
        simple_physics.CreateSiliconPotentialOnly(device, region)

        # Set up the contacts applying a bias
        for i in get_contact_list(device=device):
            if circuit_contacts and i in circuit_contacts:
                simple_physics.CreateSiliconPotentialOnlyContact(
                    device, region, i, True
                )
            else:
                set_parameter(
                    device=device, name=simple_physics.GetContactBiasName(i), value=0.0
                )
                simple_physics.CreateSiliconPotentialOnlyContact(device, region, i)

    def drift_difussion_initial_solution(self, device, region):
        # drift diffusion solution variables
        model_create.CreateSolution(device, region, "Electrons")
        model_create.CreateSolution(device, region, "Holes")

        # create initial guess from dc only solution
        set_node_values(
            device=device,
            region=region,
            name="Electrons",
            init_from="IntrinsicElectrons",
        )
        set_node_values(
            device=device, region=region, name="Holes", init_from="IntrinsicHoles"
        )

        # Set up equations
        simple_physics.CreateSiliconDriftDiffusion(device, region)
        for i in get_contact_list(device=device):
            simple_physics.CreateSiliconDriftDiffusionAtContact(device, region, i)

    def ddsolver(self) -> None:
        """Initialize mesh and solver."""
        device, self.regions, self.interfaces = create_2Duz_simulation(
            component=self.component,
            xsection_bounds=self.xsection_bounds,
            full_layerstack=self.full_layerstack,
            physical_layerstack=self.physical_layerstack,
            doping_info=self.doping_info,
            contact_info=self.contact_info,
            resolutions=self.resolutions,
            background_tag=self.background_tag,
        )
        self.device = device
        self.set_parameters(device=device, regions=self.regions)

        i = 0
        for region in self.regions["si"]:
            if i == 0:
                model_create.CreateSolution(device, region, "Potential")
                simple_physics.CreateSiliconPotentialOnly(device, region)
                i += 1
            else:
                self.initial_solution(device=device, region=region)
        for region in ["anode", "cathode"]:
            self.initial_solution(device=device, region=region)

        # model_create.CreateSolution(device, "left_clad", "Potential")
        # CreateOxidePotentialOnly(device, "left_clad")
        # self.initial_solution(device=device, region="left_clad")

        # model_create.CreateSolution(device, "right_clad", "Potential")
        # CreateOxidePotentialOnly(device, "right_clad")
        # self.initial_solution(device=device, region="right_clad")

        for region in get_region_list(device=device):
            self.drift_difussion_initial_solution(device=device, region=region)
        for interface in get_interface_list(device=device):
            simple_physics.CreateSiliconSiliconInterface(
                device=device, interface=interface
            )

        solve(
            type="dc",
            absolute_error=self.atol,
            relative_error=self.rtol,
            maximum_iterations=self.max_iter,
        )

    def ramp_voltage(self, Vfinal: float, Vstep: float, Vinit: float = 0.0) -> None:
        """Ramps the solution from Vi to Vf."""
        device = "MyDevice"
        V = Vinit
        while np.abs(V) <= np.abs(Vfinal):
            set_parameter(
                device=device,
                name=simple_physics.GetContactBiasName("cathode"),
                value=V,
            )
            solve(
                type="dc",
                absolute_error=self.atol,
                relative_error=self.rtol,
                maximum_iterations=self.max_iter,
            )
            V += Vstep


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

    # We will restrict the physical mesh to a subset of layers:
    layermap = gf.tech.LayerMap()
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
    contact_info = {}
    contact_info["anode"] = {
        "gds_layer": anode_layer,
        "physical_layerlevel_to_replace": "via_contact",
        "physical_layerlevel_to_contact": "slab90",
    }
    contact_info["cathode"] = {
        "gds_layer": cathode_layer,
        "physical_layerlevel_to_replace": "via_contact",
        "physical_layerlevel_to_contact": "slab90",
    }

    waveguide.show()

    resolutions = {}
    resolutions["core"] = {"resolution": 0.01, "distance": 2}
    resolutions["slab90"] = {"resolution": 0.03, "distance": 1}

    c = DDComponent(
        component=waveguide,
        xsection_bounds=[(4, -4), (4, 4)],
        full_layerstack=get_layer_stack_generic(),
        physical_layerstack=physical_layerstack,
        doping_info=get_doping_info_generic(),
        contact_info=contact_info,
        resolutions=resolutions,
        background_tag=None,
    )
    c.ddsolver()
    c.save_device("test.dat")
