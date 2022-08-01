"""Returns simulation from cross-section."""
import pathlib
from typing import Optional

# import matplotlib.pyplot as plt
import numpy as np

# TCAD imports
from devsim import (
    add_2d_contact,
    add_2d_interface,
    add_2d_mesh_line,
    add_2d_region,
    create_2d_mesh,
    create_device,
    finalize_mesh,
    get_contact_list,
    get_interface_list,
    get_region_list,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
)
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages.simple_physics import (
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    CreateSiliconSiliconInterface,
    GetContactBiasName,
    PrintCurrents,
    SetSiliconParameters,
)
from pydantic import BaseModel, Extra

from gdsfactory.config import CONFIG
from gdsfactory.serialization import get_hash
from gdsfactory.types import PathType

nm = 1e-9
um = 1e-6


class PINWaveguide(BaseModel):
    """Silicon PIN junction waveguide Model.

    Parameters:
        wg_width: waveguide width.
        wg_thickness: thickness waveguide (um).
        p_offset: offset between waveguide center and P-doping (um, negative to push toward n-side)
        n_offset: offset between waveguide center and N-doping (um, negative to push toward p-side)
        slab_thickness: thickness slab (um).
        t_box: thickness BOX (um).
        t_clad: thickness cladding (um).
        xmargin: margin from waveguide edge to each side (um).
        cache: filepath for caching modes. If None does not use file cache.
    ::

            xmargin          width            xmargin
           <------>       <---------->        <------>
                     ppp_offset     nnn_offset
                     <---------> <---------->
                        p_offset n_offset
                             <-> <->
                           _____|_____ _ _ _ _ _ _ _ _ _
                          |  |     |  |                |
          =====___________|  |     |  |___________=====|
          |          |       |     |         |         | wg_thickness
          |   ppp    |   p   |  i  |    n    |   nnn   |
          |__________|_______|_____|_________|_________|
          <-------------------------------------------->
                               w_sim

    """

    wg_width: float
    wg_thickness: float
    p_offset: float = 0.0
    n_offset: float = 0.0
    slab_thickness: float
    t_box: float = 2.0 * um
    t_clad: float = 2.0 * um
    xmargin: float = 1.0 * um
    cache: Optional[PathType] = CONFIG["modes"]

    class Config:
        extra = Extra.allow

    @property
    def t_sim(self):
        return self.t_box + self.wg_thickness + self.t_clad

    @property
    def w_sim(self):
        return self.wg_width + 2 * self.xmargin

    @property
    def filepath(self) -> Optional[pathlib.Path]:
        if self.cache is None:
            return
        cache = pathlib.Path(self.cache)
        cache.mkdir(exist_ok=True, parents=True)
        settings = {setting: getattr(self, setting) for setting in self.settings}
        return cache / f"{get_hash(settings)}.npz"

    def Create2DMesh(self, device):
        xmin = -self.xmargin - self.wg_width / 2
        xmax = self.xmargin + self.wg_width / 2
        ymin = 0
        ymax = self.wg_thickness
        xmin_waveguide = -self.wg_width / 2
        xmax_waveguide = self.wg_width / 2
        yslab = self.slab_thickness

        create_2d_mesh(mesh="dio")
        add_2d_mesh_line(mesh="dio", dir="x", pos=xmin, ps=100 * nm)
        add_2d_mesh_line(mesh="dio", dir="x", pos=0, ps=1 * nm)
        add_2d_mesh_line(mesh="dio", dir="x", pos=xmax, ps=100 * nm)
        add_2d_mesh_line(mesh="dio", dir="y", pos=ymin, ps=10 * nm)
        add_2d_mesh_line(mesh="dio", dir="y", pos=ymax, ps=10 * nm)

        add_2d_region(
            mesh="dio",
            material="Si",
            region="slab",
            xl=xmin,
            xh=xmax,
            yl=ymin,
            yh=yslab,
            bloat=5 * nm,
        )
        add_2d_region(
            mesh="dio",
            material="Si",
            region="core",
            xl=xmin_waveguide,
            xh=xmax_waveguide,
            yl=yslab,
            yh=ymax,
            bloat=5 * nm,
        )
        # add_2d_region(mesh="dio", material="Oxide", region="left_clad", xl=xmin, xh=xmin_waveguide, yl=yslab, yh=ymax, bloat=10*nm)
        # add_2d_region(mesh="dio", material="Oxide", region="right_clad", xl=xmax_waveguide, xh=xmax, yl=yslab, yh=ymax, bloat=10*nm)

        add_2d_interface(
            mesh="dio",
            name="i0",
            region0="core",
            region1="slab",
            xl=xmin_waveguide,
            xh=xmax_waveguide,
            yl=yslab,
            yh=yslab,
            bloat=5 * nm,
        )

        add_2d_contact(
            mesh="dio",
            name="left",
            material="metal",
            region="slab",
            yl=ymin,
            yh=yslab,
            xl=xmin,
            xh=xmin,
            bloat=1 * um,
        )
        add_2d_contact(
            mesh="dio",
            name="right",
            material="metal",
            region="slab",
            yl=ymin,
            yh=yslab,
            xl=xmax,
            xh=xmax,
            bloat=1 * um,
        )
        finalize_mesh(mesh="dio")
        create_device(mesh="dio", device=device)

    def SetParameters(self, device):
        """
        Set parameters for 300 K
        """
        SetSiliconParameters(device, "slab", 300)
        SetSiliconParameters(device, "core", 300)
        # SetOxideParameters(device, "left_clad", 300)
        # SetOxideParameters(device, "right_clad", 300)

    def SetNetDoping(self, device):
        """
        NetDoping
        """
        CreateNodeModel(device, "core", "Acceptors", "1.0e18*step(-x)")
        CreateNodeModel(device, "core", "Donors", "1.0e18*step(x)")
        CreateNodeModel(device, "core", "NetDoping", "Donors-Acceptors")
        CreateNodeModel(device, "slab", "Acceptors", "1.0e18*step(-x)")
        CreateNodeModel(device, "slab", "Donors", "1.0e18*step(x)")
        CreateNodeModel(device, "slab", "NetDoping", "Donors-Acceptors")

    def InitialSolution(self, device, region, circuit_contacts=None):
        # Create Potential, Potential@n0, Potential@n1
        CreateSolution(device, region, "Potential")

        # Create potential only physical models
        CreateSiliconPotentialOnly(device, region)

        # Set up the contacts applying a bias
        for i in get_contact_list(device=device):
            if circuit_contacts and i in circuit_contacts:
                CreateSiliconPotentialOnlyContact(device, region, i, True)
            else:
                # it is more correct for the bias to be 0, and it looks like there is side effects
                set_parameter(device=device, name=GetContactBiasName(i), value=0.0)
                CreateSiliconPotentialOnlyContact(device, region, i)

    def DriftDiffusionInitialSolution(self, device, region, circuit_contacts=None):
        # drift diffusion solution variables
        CreateSolution(device, region, "Electrons")
        CreateSolution(device, region, "Holes")

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
        CreateSiliconDriftDiffusion(device, region)
        for i in get_contact_list(device=device):
            if circuit_contacts and i in circuit_contacts:
                CreateSiliconDriftDiffusionAtContact(device, region, i, True)
            else:
                CreateSiliconDriftDiffusionAtContact(device, region, i)

    def initialize_solver(self):
        device = "MyDevice"
        self.Create2DMesh(device)
        self.SetParameters(device=device)
        self.SetNetDoping(device=device)

        CreateSolution(device, "core", "Potential")
        CreateSiliconPotentialOnly(device, "core")
        self.InitialSolution(device=device, region="slab")

        # CreateSolution(device, "left_clad", "Potential")
        # CreateOxidePotentialOnly(device, "left_clad")
        # self.InitialSolution(device=device, region="left_clad")

        # CreateSolution(device, "right_clad", "Potential")
        # CreateOxidePotentialOnly(device, "right_clad")
        # self.InitialSolution(device=device, region="right_clad")

        for region in get_region_list(device=device):
            self.DriftDiffusionInitialSolution(device=device, region=region)
        for interface in get_interface_list(device=device):
            CreateSiliconSiliconInterface(device=device, interface=interface)

        solve(
            type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=30
        )

    def ramp_voltage(self, V, dV):
        device = "MyDevice"
        v = 0.0
        while np.abs(v) < np.abs(V):
            set_parameter(device=device, name=GetContactBiasName("left"), value=v)
            solve(
                type="dc",
                absolute_error=1e10,
                relative_error=1e-10,
                maximum_iterations=30,
            )
            PrintCurrents(device, "left")
            PrintCurrents(device, "right")
            v += dV

    # def get_field_density(self, field_name="Electrons"):
    #     device = "MyDevice"
    #     region = "MyRegion"
    #     y = get_node_model_values(device=device, region=region, name=field_name)
    #     return y

    def save_device(self, filepath):
        write_devices(file=filepath, type="tecplot")


if __name__ == "__main__":
    c = PINWaveguide(
        wg_width=500 * nm,
        wg_thickness=220 * nm,
        slab_thickness=90 * nm,
    )
    c.initialize_solver()
    c.ramp_voltage(1, 0.1)

    # print(c.get_field_density(field_name="Holes"))
    c.save_device("./test")
