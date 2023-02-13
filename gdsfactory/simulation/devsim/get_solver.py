from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from devsim import (
    delete_device,
    delete_mesh,
    edge_average_model,
    edge_from_node_model,
    get_contact_list,
    get_edge_model_list,
    get_edge_model_values,
    get_interface_list,
    get_node_model_values,
    get_region_list,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
)
from devsim.python_packages import model_create, simple_physics
from pdk import get_layer_stack
from pydantic import Extra

from gdsfactory import Component
from gdsfactory.simulation.devsim.doping import (
    DopingLayerLevel,
    get_doping_info_generic,
)
from gdsfactory.simulation.devsim.get_simulation import create_2Duz_simulation
from gdsfactory.technology import LayerStack


def set_universal_parameters(device, region):
    universal = {
        "q": 1.6e-19,  # , 'coul'),
        "k": 1.3806503e-23,  # , 'J/K'),
        "Permittivity_0": 8.85e-14,  # , 'F/cm^2')
    }
    for k, v in universal.items():
        set_parameter(device=device, region=region, name=k, value=v)


class DDComponent:
    def __init__(
        self,
        component: Component,
        xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        full_layerstack: LayerStack,
        physical_layerstack: LayerStack,
        doping_info: Dict[str, DopingLayerLevel],
        contact_info=Dict,
        resolutions: Optional[Dict[str, Dict]] = None,
        mesh_scaling_factor: float = 1e-4,
        background_tag: Optional[str] = None,
        temp_file_name="temp.msh2",
        devsim_mesh_name="temp",
        devsim_device_name="temp",
        devsim_simulation_filename="devsim.dat",
        atol: float = 1e12,
        rtol: float = 1e-12,
        max_iter: int = 200,
        extended_precision: bool = True,
    ) -> None:
        """Drift-diffusion solver on a Component cross-section.

        Arguments:
            # Device parameters
            component: component
            xsection_bounds: line defined by [x1,y1], [x2,y2] to use for the meshing
            full_layerstack: complete layerstack associated with component
            physical_layerstack: layerstack subset of full_layerstack that is used to create the physical mesh
            doping_info: dict relating some full_layerstack layers and doping profiles (see doping submodule), e.g.

                doping_info = {
                    "N": DopingLayerLevel(          # key is a name (not used currently)
                        layer=layermap.N,           # which layermap layer to
                        type="Donor",               # DEVSIM dopant label, current must be either "Donor" or "Acceptor"
                        z_profile=step(n_conc),     # callable mapping dopant density to z-value
                    ),
                    "P": DopingLayerLevel(
                        layer=layermap.P,
                        type="Acceptor",
                        z_profile=step(p_conc),
                    ),

            contact_info: dict relating contact names and gds, e.g.

                contact_info = {
                    "anode": {                                              # key is contact name
                        "gds_layer": anode_layer,                           # gds layer to use for this contact
                        "physical_layerlevel_to_replace": "via_contact",    # first physical layer to use to model the contact
                        "physical_layerlevel_to_contact": "slab90",         # second physical layer, the interface with ^ defining the contact
                    },
                    "cathode": {
                        "gds_layer": cathode_layer,
                        "physical_layerlevel_to_replace": "via_contact",
                        "physical_layerlevel_to_contact": "slab90",
                    },
                }

                TODO: make this relate to only GDS layers + ports / nets

            # Mesh parameters
            resolutions: dict of mesh resolutions (see meshing module)
            mesh_scaling_factor: scales mesh dimensions (see meshing module). Defaults to um --> cm conversion
            background_tag: label of background polygon (see meshing module).

            # Filesystem parameters
            temp_file_name = temp_file_name
            devsim_mesh_name = devsim_mesh_name
            devsim_device_name = devsim_device_name
            devsim_simulation_filename = (devsim_simulation_filename,)

            # Solver parameters
            atol: absolute tolerance threshold for self-consistent solve
            rtol: relative tolerance threshold for self-consistent solve
            max_iter : maximum number of iterations for self-consistent solve
            extended_precision: if True (default), sets double precision
        """
        # Set attributes
        self.component = component
        self.xsection_bounds = xsection_bounds
        self.full_layerstack = full_layerstack or get_layer_stack()
        self.physical_layerstack = physical_layerstack
        self.doping_info = doping_info or get_doping_info_generic()
        self.contact_info = contact_info
        self.resolutions = resolutions or {}
        self.mesh_scaling_factor = mesh_scaling_factor
        self.background_tag = background_tag
        self.temp_file_name = temp_file_name
        self.devsim_mesh_name = devsim_mesh_name
        self.devsim_device_name = devsim_device_name
        self.devsim_simulation_filename = (devsim_simulation_filename,)
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter

        # Set precision
        if extended_precision:
            self.set_extended_precision()

    class Config:
        """Enable adding new."""

        extra = Extra.allow

    def set_extended_precision(self):
        set_parameter(name="extended_solver", value=True)
        set_parameter(name="extended_model", value=True)
        set_parameter(name="extended_equation", value=True)

    def set_parameters(self, region, device) -> None:
        """Set parameters for 300 K."""
        simple_physics.SetSiliconParameters(device, region, 300)

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
                # it is more correct for the bias to be 0, and it looks like there is side effects
                set_parameter(
                    device=device, name=simple_physics.GetContactBiasName(i), value=0.0
                )
                simple_physics.CreateSiliconPotentialOnlyContact(device, region, i)

    def drift_diffusion_initial_solution(self, device, region, circuit_contacts=None):
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
            if circuit_contacts and i in circuit_contacts:
                simple_physics.CreateSiliconDriftDiffusionAtContact(
                    device, region, i, True
                )
            else:
                simple_physics.CreateSiliconDriftDiffusionAtContact(device, region, i)

    def ddsolver(self, global_meshsize_array=None) -> None:
        """Initialize mesh and solver."""
        self.device, self.regions, self.interfaces = create_2Duz_simulation(
            component=self.component,
            xsection_bounds=self.xsection_bounds,
            full_layerstack=self.full_layerstack,
            physical_layerstack=self.physical_layerstack,
            doping_info=self.doping_info,
            contact_info=self.contact_info,
            resolutions=self.resolutions,
            mesh_scaling_factor=self.mesh_scaling_factor,
            background_tag=self.background_tag,
            global_meshsize_array=global_meshsize_array,
        )
        device = self.device

        for region in get_region_list(device=device):
            self.set_parameters(device=device, region=region)

        for region in get_region_list(device=device):
            self.initial_solution(device=device, region=region)

        for region in get_region_list(device=device):
            self.drift_diffusion_initial_solution(device=device, region=region)

        for interface in get_interface_list(device=device):
            simple_physics.CreateSiliconSiliconInterface(
                device=device, interface=interface
            )

        self.save_device("debug.dat")

        solve(
            type="dc",
            absolute_error=self.atol,
            relative_error=self.rtol,
            maximum_iterations=self.max_iter,
        )

    def ramp_voltage(
        self, Vfinal: float, Vstep: float, Vinit: float = 0.0, contact_name="cathode"
    ) -> None:
        """Ramps the solution from Vi to Vf."""
        V = Vinit
        while np.abs(V) <= np.abs(Vfinal):
            set_parameter(
                device=self.device,
                name=simple_physics.GetContactBiasName(contact_name),
                value=V,
            )
            solve(
                type="dc",
                absolute_error=self.atol,
                relative_error=self.rtol,
                maximum_iterations=self.max_iter,
            )
            V += Vstep

    def get_node_index(self, region_name):
        """Maps head and tail nodes of from their edge index.

        From https://github.com/devsim/devsim_misc/blob/9a3c7056e0e3e7fc49e17031a706573350292d4d/refinement/refinement2.py#L45
        """
        if "node_index@n0" not in get_edge_model_list(
            device=self.device, region=region_name
        ):
            edge_from_node_model(
                node_model="node_index", device=self.device, region=region_name
            )
        return list(
            zip(
                [
                    int(x)
                    for x in get_edge_model_values(
                        device=self.device,
                        region=region_name,
                        name="node_index@n0",
                    )
                ],
                [
                    int(x)
                    for x in get_edge_model_values(
                        device=self.device,
                        region=region_name,
                        name="node_index@n1",
                    )
                ],
            )
        )

    def get_node_field(self, region_name, field_name="Electrons"):
        return get_node_model_values(
            device=self.device, region=region_name, name=field_name
        )

    def get_mean_edge_from_node_field(self, region_name, node_field="x"):
        edge_average_model(
            device=self.device,
            region=region_name,
            node_model=node_field,
            edge_model=f"{node_field}@mean",
            average_type="arithmetic",
        )
        return get_edge_model_values(
            device=self.device, region=region_name, name=f"{node_field}@mean"
        )

    def get_edge_field(self, region_name, field_name="EdgeLength"):
        return get_edge_model_values(
            device=self.device, region=region_name, name=field_name
        )

    def get_regions(self):
        return get_region_list(device=self.device)

    def save_device(self, filepath) -> None:
        """Save Device to a tecplot filepath that you can open with Paraview."""
        write_devices(file=filepath, type="tecplot")

    def delete_device(self) -> None:
        """Delete this devsim device and mesh."""
        delete_mesh(mesh=self.devsim_mesh_name)
        delete_device(device=self.devsim_mesh_name)

    def get_refined_mesh(
        self,
        factor: float = 2.0,
        refine_dict: Dict[str, Dict] = None,
        refine_regions: Tuple[str, ...] = ("si",),
    ):
        """Refines the mesh based on simulation result.

        Currently only remeshes in silicon regions.

        Arguments:
            factor: number to scale the mesh characteristic length down where refine_dict is True.
            refine_dict: Dict of fields:conditions to determine where to remesh, e.g.
                refine_dict = {
                    "Potential": {                                  # top-level key is field name
                        "func": lambda x0, x1: abs(x0 - x1),        # callable to apply to the local field difference before checking threshold
                        "threshold": 0.05,                          # will remesh if func(field) > threshold
                    },
                    "Electrons": {
                        "func": lambda x0, x1: np.abs(np.log10(x0) - np.log10(x1)),
                        "threshold": 1,
                    },
                }
            refine_regions: region keys where to refine.
        """
        xs = {}
        ys = {}
        lcs = {}
        refine = {}

        # Refined regions
        for regiontype, region_names in c.regions.items():
            for regionname in region_names:
                if regiontype in refine_regions:
                    xs[regionname] = np.array(
                        c.get_mean_edge_from_node_field(regionname, "x")
                    )
                    ys[regionname] = np.array(
                        c.get_mean_edge_from_node_field(regionname, "y")
                    )
                    lcs[regionname] = c.get_edge_field(
                        regionname, field_name="EdgeLength"
                    )
                    node_index = c.get_node_index(region_name=regionname)

                    refinements = []
                    for field_name, refinement in refine_dict.items():
                        field = np.array(c.get_node_field(regionname, field_name))
                        refinements.append(
                            np.array(
                                [
                                    refinement["func"](field[x[0]], field[x[1]])
                                    > refinement["threshold"]
                                    for x in node_index
                                ]
                            )
                        )

                    refine[regionname] = refinements[0]
                    for index in range(1, len(refinements)):
                        refine[regionname] = np.logical_or(
                            refinements[index], refinements[index - 1]
                        )

        um_to_cm = 1e-4
        xs = np.hstack([np.array(x) for x in xs.values()], dtype=np.float64) / um_to_cm
        ys = np.hstack([np.array(y) for y in ys.values()], dtype=np.float64) / um_to_cm
        lcs = (
            np.hstack([np.array(lc) for lc in lcs.values()], dtype=np.float64)
            / um_to_cm
        )
        refine = np.hstack([np.array(x) for x in refine.values()], dtype=bool)

        lcs_after = np.where(refine, lcs / factor, lcs)

        N = len(xs)
        meshing_field = np.zeros([N, 4])
        meshing_field[:, 0] = xs
        meshing_field[:, 1] = ys
        meshing_field[:, 2] = np.zeros(N)
        meshing_field[:, 3] = lcs_after
        return meshing_field


if __name__ == "__main__":
    import gdsfactory as gf

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
            k: get_layer_stack().layers[k]
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

    # Initial meshing
    resolutions = {
        "core": {"resolution": 0.02, "distance": 1},
        "slab90": {"resolution": 0.02, "distance": 1},
    }

    c = DDComponent(
        component=waveguide,
        xsection_bounds=[(4, -4), (4, 4)],
        full_layerstack=get_layer_stack(),
        physical_layerstack=physical_layerstack,
        doping_info=get_doping_info_generic(),
        contact_info=contact_info,
        resolutions=resolutions,
        mesh_scaling_factor=1e-4,
        background_tag=None,
    )

    c.ddsolver()
    c.save_device("test1.dat")

    meshing_field = c.get_refined_mesh(
        factor=2.0,
        refine_dict={
            "Potential": {
                "func": lambda x0, x1: abs(x0 - x1),
                "threshold": 0.05,
            },
            "Electrons": {
                "func": lambda x0, x1: np.abs(np.log10(x0) - np.log10(x1)),
                "threshold": 1,
            },
        },
        refine_regions=["si"],
    )

    c.delete_device()
    c = DDComponent(
        component=waveguide,
        xsection_bounds=[(4, -4), (4, 4)],
        full_layerstack=get_layer_stack(),
        physical_layerstack=physical_layerstack,
        doping_info=get_doping_info_generic(),
        contact_info=contact_info,
        resolutions=resolutions,
        mesh_scaling_factor=1e-4,
        background_tag=None,
    )

    c.ddsolver(global_meshsize_array=meshing_field)
    c.save_device("test2.dat")
