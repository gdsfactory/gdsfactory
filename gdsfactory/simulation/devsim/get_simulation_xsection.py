"""Returns simulation from cross-section."""

import devsim
import numpy as np
import pyvista as pv
from devsim.python_packages import model_create, simple_physics
from pydantic import BaseModel, Extra
from wurlitzer import pipes

import matplotlib.pyplot as plt

from gdsfactory.config import CONFIG, logger
import pathlib
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union
from gdsfactory.config import CONFIG, logger
from gdsfactory.serialization import get_hash
from gdsfactory.simulation.gtidy3d.materials import si, sin, sio2
from gdsfactory.types import PathType, TypedArray

from gdsfactory.simulation.gtidy3d.modes import Precision, FilterPol, Waveguide

import pdb 

nm = 1e-9
um = 1e-6

"""
Phenomenological wavelength-dependent index and absorption perturbation from free carriers
Use quadratic fits for wavelengths, or better-characterized models at 1550 and 1310

From Chrostowski, L., & Hochberg, M. (2015). Silicon Photonics Design: From Devices to Systems. Cambridge University Press. doi: 10.1017/CBO9781316084168
Citing:
(1) R. Soref and B. Bennett, "Electrooptical effects in silicon," in IEEE Journal of Quantum Electronics, vol. 23, no. 1, pp. 123-129, January 1987, doi: 10.1109/JQE.1987.1073206.
(2) Reed, G. T., Mashanovich, G., Gardes, F. Y., & Thomson, D. J. (2010). Silicon optical modulators. Nature Photonics, 4(8), 518–526. doi: 10.1038/nphoton.2010.179
(3) M. Nedeljkovic, R. Soref and G. Z. Mashanovich, "Free-Carrier Electrorefraction and Electroabsorption Modulation Predictions for Silicon Over the 1–14- $\mu\hbox{m}$ Infrared Wavelength Range," in IEEE Photonics Journal, vol. 3, no. 6, pp. 1171-1180, Dec. 2011, doi: 10.1109/JPHOT.2011.2171930.

Parameters:
    wavelength: (um)
    dN: excess electrons (/cm^3)
    dP: excess holes (/cm^3)

Returns:
    dn: change in refractive index
    or
    dalpha: change in absorption coefficient (/cm)
"""

def dn_carriers(wavelength, dN, dP):
    if wavelength == 1.55:
        return -5.4*1E-22*np.power(dN, 1.011) - 1.53*1E-18*np.power(dP, 0.838)
    elif wavelength == 1.31:
        return -2.98*1E-22*np.power(dN, 1.016) - 1.25*1E-18*np.power(dP, 0.835)
    else:
        wavelength = wavelength * 1E-6 # convert to m
        return -3.64*1E-10*wavelength**2*dN - 3.51*1E-6*wavelength**2*np.poewr(dP, 0.8)

def dalpha_carriers(wavelength, dN, dP):
    if wavelength == 1.55:
        return 8.88*1E-21*dN**1.167 + 5.84*1E-20*dP**1.109
    elif wavelength == 1.31:
        return 3.48*1E-22*dN**1.229 + 1.02*1E-19*dP**1.089
    else:
        wavelength = wavelength * 1E-6 # convert to m
        return 3.52*1E-6*wavelength**2*dN + 2.4*1E-6*wavelength**2*dP

class PINWaveguide(BaseModel):
    """Silicon PIN junction waveguide Model.

    Parameters:
        wg_width: waveguide width.
        wg_thickness: thickness waveguide (um).
        p_offset: offset between waveguide center and P-doping in um
            negative to push toward n-side.
        n_offset: offset between waveguide center and N-doping in um
            negative to push toward p-side).
        ppp_offset: offset between waveguide center and Ppp-doping in um
            negative to push toward n-side) NOT IMPLEMENTED.
        npp_offset: offset between waveguide center and Npp-doping in um
            negative to push toward p-side) NOT IMPLEMENTED.
        slab_thickness: thickness slab (um).
        t_box: thickness BOX (um).
        t_clad: thickness cladding (um).
        p_conc: low-doping acceptor concentration (/cm3).
        n_conc: low-doping donor concentration (/cm3).
        ppp_conc: high-doping acceptor concentration (/cm3).
        nnn_conc: high-doping donor concentration (/cm3).
        xmargin: margin from waveguide edge to low doping regions.
        contact_bloat: controls which nodes are considered contacts;
            adjust if contacts are not found.
        atol: tolerance for iterative solver
        rtol: tolerance for iterative solver
        max_iter: maximum number of iterations of iterative solver

    ::

            xmargin          width            xmargin
           <------>       <---------->        <------>
                   ppp_offset        npp_offset
                     <--->             <---->
                        p_offset n_offset
                             <-> <->
         xcontact          _____|_____ _ _ _ _ _ _ _ _ _
          <---->          |  |     |  |                |
          ======__________|  |     |  |__________======|
          |          |       |     |         |         | wg_thickness
          |  Ppp+P   |   P   |  i  |    N    |  Npp+N  |
          |__________|_______|_____|_________|_________|
          ^          ^       ^     ^         ^         ^
      pp_res_x   pp_p_res     pn_res      pp_p_res  pp_res_x
          <-------------------------------------------->
                               w_sim

    """

    wg_width: float
    wg_thickness: float
    p_offset: float = 0.0
    n_offset: float = 0.0
    ppp_offset: float = 0.5 * um
    npp_offset: float = 0.5 * um
    slab_thickness: float
    t_box: float = 2.0 * um
    t_clad: float = 2.0 * um
    p_conc: float = 1e18
    n_conc: float = 1e18
    ppp_conc: float = 1e20
    nnn_conc: float = 1e20
    xmargin: float = 0.5 * um
    xcontact: float = 0.25 * um
    pp_res_x: float = 20 * nm
    pp_p_res_x: float = 5 * nm
    p_res_x: float = 2 * nm
    pn_res_x: float = 1 * nm
    coarse_res_y: float = 10 * nm
    slab_res_y: float = 2 * nm
    atol: float = 1E8
    rtol: float = 1E-8
    max_iter: int = 60

    class Config:
        """Enable adding new."""

        extra = Extra.allow

    @property
    def t_sim(self):
        return self.t_box + self.wg_thickness + self.t_clad

    @property
    def w_sim(self):
        return 2 * self.xmargin + self.ppp_offset + self.npp_offset

    def Create2DMesh(self, device):
        xmin = -self.xmargin - self.ppp_offset - self.wg_width / 2
        xmax = self.xmargin + self.npp_offset + self.wg_width / 2
        self.xppp = -self.ppp_offset - self.wg_width / 2
        self.xnpp = self.npp_offset + self.wg_width / 2
        ymin = 0
        ymax = self.wg_thickness
        xmin_waveguide = -self.wg_width / 2
        xmax_waveguide = self.wg_width / 2
        yslab = self.slab_thickness

        devsim.create_2d_mesh(mesh="dio")
        devsim.add_2d_mesh_line(mesh="dio", dir="x", pos=xmin, ps=self.pp_res_x)
        devsim.add_2d_mesh_line(mesh="dio", dir="x", pos=self.xppp, ps=self.pp_p_res_x)
        devsim.add_2d_mesh_line(mesh="dio", dir="x", pos=self.xppp / 2, ps=self.p_res_x)
        devsim.add_2d_mesh_line(
            mesh="dio", dir="x", pos=-self.p_offset, ps=self.pn_res_x
        )
        devsim.add_2d_mesh_line(
            mesh="dio", dir="x", pos=self.n_offset, ps=self.pn_res_x
        )
        devsim.add_2d_mesh_line(mesh="dio", dir="x", pos=self.xnpp / 2, ps=self.p_res_x)
        devsim.add_2d_mesh_line(mesh="dio", dir="x", pos=self.xnpp, ps=self.pp_p_res_x)
        devsim.add_2d_mesh_line(mesh="dio", dir="x", pos=xmax, ps=self.pp_res_x)
        devsim.add_2d_mesh_line(mesh="dio", dir="y", pos=ymin, ps=self.coarse_res_y)
        devsim.add_2d_mesh_line(mesh="dio", dir="y", pos=yslab, ps=self.slab_res_y)
        devsim.add_2d_mesh_line(mesh="dio", dir="y", pos=ymax, ps=self.coarse_res_y)

        devsim.add_2d_region(
            mesh="dio",
            material="Si",
            region="slab",
            xl=xmin,
            xh=xmax,
            yl=ymin,
            yh=yslab,
        )
        devsim.add_2d_region(
            mesh="dio",
            material="Si",
            region="core",
            xl=xmin_waveguide,
            xh=xmax_waveguide,
            yl=yslab,
            yh=ymax,
        )
        devsim.add_2d_region(
            mesh="dio",
            material="Si",
            region="left_contact",
            xl=xmin,
            xh=xmin + self.xcontact,
            yl=yslab,
            yh=yslab + 1 * nm,
        )
        devsim.add_2d_region(
            mesh="dio",
            material="Si",
            region="right_contact",
            xl=xmax - self.xcontact,
            xh=xmax,
            yl=yslab,
            yh=yslab + 1 * nm,
        )

        devsim.add_2d_interface(
            mesh="dio",
            name="i0",
            region0="core",
            region1="slab",
            xl=xmin_waveguide,
            xh=xmax_waveguide,
            yl=yslab,
            yh=yslab,
        )

        devsim.add_2d_contact(
            mesh="dio",
            name="left",
            material="metal",
            region="slab",
            yl=ymin,
            yh=yslab,
            xl=xmin,
            xh=xmin + self.xmargin / 2,
        )
        devsim.add_2d_contact(
            mesh="dio",
            name="right",
            material="metal",
            region="slab",
            yl=ymin,
            yh=yslab,
            xl=xmax,
            xh=xmax - self.xmargin / 2,
        )
        devsim.finalize_mesh(mesh="dio")
        devsim.create_device(mesh="dio", device=device)

    def SetParameters(self, device):
        """Set parameters for 300 K."""
        simple_physics.SetSiliconParameters(device, "slab", 300)
        simple_physics.SetSiliconParameters(device, "core", 300)
        simple_physics.SetSiliconParameters(device, "left_contact", 300)
        simple_physics.SetSiliconParameters(device, "right_contact", 300)

    def SetNetDoping(self, device):
        """NetDoping."""
        model_create.CreateNodeModel(
            device,
            "slab",
            "Acceptors",
            f"{self.p_conc:1.3e}*step({-1*self.p_offset:1.3e}-x) + {self.ppp_conc:1.3e}*step({self.xppp:1.3e}-x)",
        )
        model_create.CreateNodeModel(
            device,
            "slab",
            "Donors",
            f"{self.n_conc:1.3e}*step(x-{self.n_offset:1.3e}) + {self.nnn_conc:1.3e}*step(x-{self.xnpp:1.3e})",
        )
        model_create.CreateNodeModel(device, "slab", "NetDoping", "Donors-Acceptors")
        model_create.CreateNodeModel(
            device,
            "core",
            "Acceptors",
            f"{self.p_conc:1.1e}*step({-1*self.p_offset:1.3e}-x)",
        )
        model_create.CreateNodeModel(
            device, "core", "Donors", f"{self.n_conc:1.1e}*step(x-{self.n_offset:1.3e})"
        )
        model_create.CreateNodeModel(device, "core", "NetDoping", "Donors-Acceptors")
        model_create.CreateNodeModel(device, "left_contact", "NetDoping", "0")
        model_create.CreateNodeModel(device, "right_contact", "NetDoping", "0")

    def InitialSolution(self, device, region, circuit_contacts=None):
        # Create Potential, Potential@n0, Potential@n1
        model_create.CreateSolution(device, region, "Potential")

        # Create potential only physical models
        simple_physics.CreateSiliconPotentialOnly(device, region)

        # Set up the contacts applying a bias
        for i in devsim.get_contact_list(device=device):
            if circuit_contacts and i in circuit_contacts:
                simple_physics.CreateSiliconPotentialOnlyContact(
                    device, region, i, True
                )
            else:
                # it is more correct for the bias to be 0, and it looks like there is side effects
                devsim.set_parameter(
                    device=device, name=simple_physics.GetContactBiasName(i), value=0.0
                )
                simple_physics.CreateSiliconPotentialOnlyContact(device, region, i)

    def DriftDiffusionInitialSolution(self, device, region, circuit_contacts=None):
        # drift diffusion solution variables
        model_create.CreateSolution(device, region, "Electrons")
        model_create.CreateSolution(device, region, "Holes")

        # create initial guess from dc only solution
        devsim.set_node_values(
            device=device,
            region=region,
            name="Electrons",
            init_from="IntrinsicElectrons",
        )
        devsim.set_node_values(
            device=device, region=region, name="Holes", init_from="IntrinsicHoles"
        )

        # Set up equations
        simple_physics.CreateSiliconDriftDiffusion(device, region)
        for i in devsim.get_contact_list(device=device):
            if circuit_contacts and i in circuit_contacts:
                simple_physics.CreateSiliconDriftDiffusionAtContact(
                    device, region, i, True
                )
            else:
                simple_physics.CreateSiliconDriftDiffusionAtContact(device, region, i)

    def ddsolver(self):
        device = "MyDevice"
        self.Create2DMesh(device)
        self.SetParameters(device=device)
        self.SetNetDoping(device=device)

        model_create.CreateSolution(device, "core", "Potential")
        simple_physics.CreateSiliconPotentialOnly(device, "core")
        self.InitialSolution(device=device, region="slab")
        self.InitialSolution(device=device, region="right_contact")
        self.InitialSolution(device=device, region="left_contact")

        # model_create.CreateSolution(device, "left_clad", "Potential")
        # CreateOxidePotentialOnly(device, "left_clad")
        # self.InitialSolution(device=device, region="left_clad")

        # model_create.CreateSolution(device, "right_clad", "Potential")
        # CreateOxidePotentialOnly(device, "right_clad")
        # self.InitialSolution(device=device, region="right_clad")

        for region in devsim.get_region_list(device=device):
            self.DriftDiffusionInitialSolution(device=device, region=region)
        for interface in devsim.get_interface_list(device=device):
            simple_physics.CreateSiliconSiliconInterface(
                device=device, interface=interface
            )

        devsim.solve(
            type="dc", absolute_error=self.atol, relative_error=self.rtol, maximum_iterations=self.max_iter
        )

    def ramp_voltage(self, V, dV):
        device = "MyDevice"
        v = 0.0
        while np.abs(v) <= np.abs(V):
            devsim.set_parameter(
                device=device, name=simple_physics.GetContactBiasName("left"), value=v
            )
            devsim.solve(
                type="dc",
                absolute_error=self.atol,
                relative_error=self.rtol,
                maximum_iterations=self.max_iter,
            )
            # simple_physics.PrintCurrents(device, "left")
            # simple_physics.PrintCurrents(device, "right")
            v += dV

    def get_field(self, region_name="core", field_name="Electrons"):
        device = "MyDevice"
        y = devsim.get_node_model_values(device=device, region=region_name, name=field_name)
        return y

    def save_device(self, filepath):
        devsim.write_devices(file=filepath, type="tecplot")

    def plot(
        self,
        tempfile="temp.dat",
        scalars=None,
        log_scale=False,
        cmap="RdBu",
        jupyter_backend="None",
    ):
        devsim.write_devices(file=tempfile, type="tecplot")
        reader = pv.get_reader(tempfile)
        mesh = reader.read()
        # sargs = dict(height=0.25, vertical=True, position_x=0.05, position_y=0.05)
        plotter = pv.Plotter(notebook=True)
        _ = plotter.add_mesh(
            mesh, scalars=scalars, log_scale=log_scale, cmap=cmap
        )  # , scalar_bar_args=sargs)
        _ = plotter.show_grid()
        _ = plotter.camera_position = "xy"
        with pipes() as (out, err):
            _ = plotter.show(jupyter_backend=jupyter_backend)

    def list_fields(self, tempfile="temp.dat"):
        devsim.write_devices(file=tempfile, type="tecplot")
        reader = pv.get_reader(tempfile)
        mesh = reader.read()
        return mesh[0]


if __name__ == "__main__":
    c = PINWaveguide(
        wg_width=500 * nm,
        wg_thickness=220 * nm,
        slab_thickness=90 * nm,
    )
    c.ddsolver()
    # c.save_device("./test.dat")
    c.ramp_voltage(1.0, 0.1)

    # # print(c.get_field_density(field_name="Holes"))
    c.save_device("./test.dat")
