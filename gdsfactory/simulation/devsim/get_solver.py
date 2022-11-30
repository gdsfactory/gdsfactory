from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from devsim import (
    delete_device,
    delete_mesh,
    get_contact_list,
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
from pydantic import BaseModel, Extra

from gdsfactory import Component
from gdsfactory.simulation.devsim.doping import get_doping_info_generic
from gdsfactory.simulation.devsim.get_simulation import create_2Duz_simulation
from gdsfactory.tech import LayerStack


def set_universal_parameters(device, region):
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
    mesh_scaling_factor: float = (1.0,)
    background_tag: Optional[str] = None
    temp_file_name = "temp.msh2"
    devsim_mesh_name = "temp"
    devsim_device_name = "temp"
    devsim_simulation_filename = "devsim.dat"
    atol: float = 1e8
    rtol: float = 1e-8
    max_iter: int = 100

    class Config:
        """Enable adding new."""

        extra = Extra.allow

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

    def get_node_field(self, region_name, field_name="Electrons"):
        return get_node_model_values(
            device=self.device, region=region_name, name=field_name
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


if __name__ == "__main__":

    import gdsfactory as gf
    from gdsfactory.tech import get_layer_stack_generic

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
        "slab90": {"resolution": 0.05, "distance": 1},
    }

    c = DDComponent(
        component=waveguide,
        xsection_bounds=[(4, -4), (4, 4)],
        full_layerstack=get_layer_stack_generic(),
        physical_layerstack=physical_layerstack,
        doping_info=get_doping_info_generic(),
        contact_info=contact_info,
        resolutions=resolutions,
        mesh_scaling_factor=1e-4,
        background_tag=None,
    )

    c.ddsolver()
    c.save_device("test1.dat")

    # c.ramp_voltage(-0.3, -0.1, contact_name="cathode")
    # c.save_device("test2.dat")

    # Tabulate old lcs
    xs = {}
    ys = {}
    lcs = {}
    potentials = {}
    electrons = {}
    holes = {}
    for regionname in c.get_regions():
        xs[regionname] = np.array(c.get_node_field(regionname, "x"))
        ys[regionname] = np.array(c.get_node_field(regionname, "y"))
        lcs[regionname] = np.sqrt(np.array(c.get_node_field(regionname, "NodeVolume")))
        potentials[regionname] = np.array(c.get_node_field(regionname, "Potential"))
        electrons[regionname] = np.array(c.get_node_field(regionname, "Electrons"))
        holes[regionname] = np.array(c.get_node_field(regionname, "Holes"))

    N = sum(len(x) for x in xs.values())
    # old_lcs = np.zeros([N, 4])
    um_to_cm = 1e-4
    xs = np.hstack([np.array(x) for x in xs.values()]) / um_to_cm
    ys = np.hstack([np.array(y) for y in ys.values()]) / um_to_cm
    lcs = np.hstack([np.array(lc) for lc in lcs.values()]) / um_to_cm
    potentials = np.hstack([np.array(x) for x in potentials.values()])
    electrons = np.hstack([np.array(x) for x in electrons.values()])
    holes = np.hstack([np.array(x) for x in holes.values()])

    # Get gradient of electron and potential fields
    from scipy.interpolate import SmoothBivariateSpline

    potential_g = SmoothBivariateSpline(x=xs, y=ys, z=potentials).partial_derivative(
        dx=1, dy=1
    )
    electrons_g = SmoothBivariateSpline(x=xs, y=ys, z=electrons).partial_derivative(
        dx=1, dy=1
    )
    holes_g = SmoothBivariateSpline(x=xs, y=ys, z=holes).partial_derivative(dx=1, dy=1)

    # Use gradient to reduce characteristic length in fast-varying regions
    import matplotlib.pyplot as plt

    # plt.plot(grid_z2(xs, ys, grid=False))

    plt.scatter(xs, lcs)
    plt.show()
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(
        xs,
        np.abs(potential_g(xs, ys, grid=False))
        / np.max(np.abs(potential_g(xs, ys, grid=False))),
    )
    ax.set_yscale("log")
    plt.show()
    ax = plt.gca()
    ax.scatter(
        xs,
        np.abs(electrons_g(xs, ys, grid=False))
        / np.max(np.abs(electrons_g(xs, ys, grid=False))),
    )
    ax.scatter(
        xs,
        np.abs(holes_g(xs, ys, grid=False))
        / np.max(np.abs(holes_g(xs, ys, grid=False))),
    )
    ax.set_yscale("log")
    plt.show()

    def rescale(lcs, field, xs, ys, step=0.5):
        """Rescale lcs depending on field."""
        # inds = np.where(xs > xlims[0] and xs < xlims[1] and ys > ylims[0] and ys < ylims[1])
        norm = np.abs(field(xs, ys, grid=False)) / np.max(
            np.abs(field(xs, ys, grid=False))
        )
        return lcs * (1 - step * norm)

    # um_to_cm = 1e-4
    meshing_field = np.zeros([N, 4])
    meshing_field[:, 0] = xs
    meshing_field[:, 1] = ys
    meshing_field[:, 2] = np.zeros(N)
    meshing_field[:, 3] = (
        (
            rescale(lcs, electrons_g, xs, ys, step=0.02)
            + rescale(lcs, holes_g, xs, ys, step=0.02)
        )
        / 2
        * rescale(lcs, potential_g, xs, ys, step=0.02)
    )

    print(np.shape(meshing_field))
    print(meshing_field[:, 0])
    print(meshing_field[:, 1])
    print(meshing_field[:, 2])
    print(meshing_field[:, 3])

    plt.scatter(meshing_field[:, 0], lcs)
    plt.scatter(meshing_field[:, 0], meshing_field[:, 3])
    plt.show()

    # c.delete_device()
    # c = DDComponent(
    #     component=waveguide,
    #     xsection_bounds=[(4, -4), (4, 4)],
    #     full_layerstack=get_layer_stack_generic(),
    #     physical_layerstack=physical_layerstack,
    #     doping_info=get_doping_info_generic(),
    #     contact_info=contact_info,
    #     resolutions=resolutions,
    #     mesh_scaling_factor=1e-4,
    #     background_tag=None,
    # )

    # c.ddsolver(global_meshsize_array=meshing_field)
    # c.save_device("test_remeshed.dat")

    # Get gradients to remesh on
    # import pyvista as pv
    # reader = pv.get_reader('test1.dat')
    # mesh = reader.read()

    # x_g = []
    # y_g = []

    # potential_g = []
    # electrons_g = []
    # for block in mesh:
    #     potential_g.append(block.compute_derivative(scalars="Potential")["gradient"])
    #     electrons_g.append(block.compute_derivative(scalars="Electrons")["gradient"])
    # potential_g = np.vstack(potential_g)
    # electrons_g = np.vstack(electrons_g)
    # # potential_g = np.insert(np.vstack(potential_g), 2, values=0, axis=1)
    # # electrons_g = np.insert(np.vstack(electrons_g), 2, values=0, axis=1)

    # # If gradients are large at a given node, reduce its characteristic length
    # import matplotlib.pyplot as plt
    # plt.plot(electrons)
    # plt.show()
    # print(np.max(np.abs(potential_g[:,2])))
    # print(np.max(np.abs(electrons_g[:,2])))

    # c.delete_device()
    # c.ddsolver(global_meshsize_array=mesh_g)
    # c.save_device("test2.dat")

    # c.ramp_voltage(-2, -0.5, contact_name="cathode")
    # c.save_device("test2.dat")

    # xs = {}
    # ys = {}
    # efields = {}
    # edensities = {}
    # for regionname in c.regions["si"]:
    #     xs[regionname] = np.array(c.get_node_field(regionname, "x"))
    #     ys[regionname] = np.array(c.get_node_field(regionname, "y"))
    #     edensities[regionname] = np.array(np.log10(c.get_node_field(regionname, "Electrons")))
    # for regionname in c.regions["si"]:
    #     efields[regionname] = np.array(c.get_edge_field(regionname, "ElectricField")) # derivative of potential

    # c.save_device("test0.dat")
    # c.delete_device()

    # # Second, refined meshing
    # lc_min = 0.001
    # lc_max = 0.2
    # N = sum(len(x) for x in xs.values())
    # print(N)
    # mesh_g = (lc_max + lc_min)/2*np.ones([N,4])
    # mesh_g[:,0] = np.hstack([np.array(x) for x in xs.values()])
    # mesh_g[:,1] = np.hstack([np.array(y) for y in ys.values()])
    # mesh_g[:,2] = np.zeros(N)

    # def rescale(arr):
    #     arr /= np.max(np.abs(arr)) # now 0 to 1
    #     arr = np.abs(arr) # now 0 to 1
    #     arr = np.reciprocal(arr) # map from 1 to infty
    #     return np.reciprocal(arr/np.max(np.abs(arr),axis=0), )

    # mesh_g[:,3] = np.reciprocal(np.abs(np.hstack([np.array(efield) for efield in efields.values()])))
    # print(mesh_g)
