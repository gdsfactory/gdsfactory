r"""Returns simulation from cross-section.

From Chrostowski, L., & Hochberg, M. (2015). Silicon Photonics Design: From Devices to Systems. Cambridge University Press. doi: 10.1017/CBO9781316084168
    Citing:
    (1) R. Soref and B. Bennett, "Electrooptical effects in silicon," in IEEE Journal of Quantum Electronics, vol. 23, no. 1, pp. 123-129, January 1987, doi: 10.1109/JQE.1987.1073206.
    (2) Reed, G. T., Mashanovich, G., Gardes, F. Y., & Thomson, D. J. (2010). Silicon optical modulators. Nature Photonics, 4(8), 518–526. doi: 10.1038/nphoton.2010.179
    (3) M. Nedeljkovic, R. Soref and G. Z. Mashanovich, "Free-Carrier Electrorefraction and Electroabsorption Modulation Predictions for Silicon Over the 1–14- $\mu\hbox{m}$ Infrared Wavelength Range," in IEEE Photonics Journal, vol. 3, no. 6, pp. 1171-1180, Dec. 2011, doi: 10.1109/JPHOT.2011.2171930.

"""


from __future__ import annotations

import warnings
from typing import Optional, TYPE_CHECKING

import devsim
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import tidy3d as td
from devsim.python_packages import model_create, simple_physics
from pydantic import BaseModel, Extra
from scipy.interpolate import griddata

from gdsfactory.simulation.disable_print import disable_print, enable_print
from gdsfactory.simulation.gtidy3d.materials import get_nk
from gdsfactory.simulation.gtidy3d.modes import Precision, Waveguide

if TYPE_CHECKING:
    from gdsfactory.typings import MaterialSpec

nm = 1e-9
um = 1e-6
cm = 1e-2

DEVSIM_INTERPOLATION_METHOD = "linear"


def dn_carriers(wavelength: float, dN: float, dP: float) -> float:
    """Phenomenological wavelength-dependent index perturbation from free carriers.

    Use quadratic fits for wavelengths, or better-characterized models at 1550 and 1310.

    Args:
        wavelength: (um).
        dN: excess electrons (/cm^3).
        dP: excess holes (/cm^3).

    Returns:
        dalpha: change in absorption coefficient (/cm)
    """
    if wavelength == 1.55:
        return -5.4 * 1e-22 * np.power(dN, 1.011) - 1.53 * 1e-18 * np.power(dP, 0.838)
    elif wavelength == 1.31:
        return -2.98 * 1e-22 * np.power(dN, 1.016) - 1.25 * 1e-18 * np.power(dP, 0.835)
    else:
        wavelength *= 1e-6
        return (
            -3.64 * 1e-10 * wavelength**2 * dN
            - 3.51 * 1e-6 * wavelength**2 * np.power(dP, 0.8)
        )


def dalpha_carriers(wavelength: float, dN: float, dP: float) -> float:
    """Phenomenological wavelength-dependent absorption perturbation from free carriers.

    Use quadratic fits for wavelengths, or better-characterized models at 1550 and 1310.

    Args:
        wavelength: (um).
        dN: excess electrons (/cm^3).
        dP: excess holes (/cm^3).

    Returns:
        dalpha: change of absorption coefficient (/cm).
    """
    if wavelength == 1.55:
        return 8.88 * 1e-21 * dN**1.167 + 5.84 * 1e-20 * dP**1.109
    elif wavelength == 1.31:
        return 3.48 * 1e-22 * dN**1.229 + 1.02 * 1e-19 * dP**1.089
    else:
        wavelength *= 1e-6
        return 3.52 * 1e-6 * wavelength**2 * dN + 2.4 * 1e-6 * wavelength**2 * dP


def alpha_to_k(alpha, wavelength):
    """Converts absorption coefficient (/cm) to extinction coefficient (unitless), given wavelength (um)."""
    wavelength = wavelength * 1e-6  # convert to m
    alpha = alpha * 1e2  # convert to /m
    return alpha * wavelength / (4 * np.pi)


def k_to_alpha(k, wavelength):
    """Converts extinction coefficient (unitless) to absorption coefficient (/cm), given wavelength (um)."""
    wavelength = wavelength * 1e-6  # convert to m
    alpha = 4 * np.pi * k / wavelength
    return alpha * 1e-2  # convert to /cm


class PINWaveguide(BaseModel):
    """Silicon PIN junction waveguide Model.

    Parameters:
        core_width: waveguide width.
        core_thickness: thickness waveguide (um).
        p_offset: offset between waveguide center and P-doping in um
            negative to push toward n-side.
        n_offset: offset between waveguide center and N-doping in um
            negative to push toward p-side).
        ppp_offset: offset between waveguide center and Ppp-doping in um
            negative to push toward n-side) NOT IMPLEMENTED.
        npp_offset: offset between waveguide center and Npp-doping in um
            negative to push toward p-side) NOT IMPLEMENTED.
        slab_thickness: thickness slab (um).
        box_thickness: thickness BOX (um).
        clad_thickness: thickness cladding (um).
        p_conc: low-doping acceptor concentration (/cm3).
        n_conc: low-doping donor concentration (/cm3).
        ppp_conc: high-doping acceptor concentration (/cm3).
        nnn_conc: high-doping donor concentration (/cm3).
        xmargin: margin from waveguide edge to low doping regions.
        contact_bloat: controls which nodes are considered contacts;
            adjust if contacts are not found.
        atol: tolerance for iterative solver.
        rtol: tolerance for iterative solver.
        max_iter: maximum number of iterations of iterative solver.

    ::

            xmargin          width            xmargin
           <------>       <---------->        <------>
                   ppp_offset        npp_offset
                     <--->             <---->
                        p_offset n_offset
                             <-> <->
         xcontact          _____|_____ _ _ _ _ _ _ _ _ _
          <---->          |  |     |  |                | core_thickness
          ======__________|  |     |  |__________======| _
          |          |       |     |         |         | |
          |  Ppp+P   |   P   |  i  |    N    |  Npp+N  | | slab_thickness
          |__________|_______|_____|_________|_________| |
          ^          ^       ^     ^         ^         ^
      pp_res_x   pp_p_res     pn_res      pp_p_res  pp_res_x
          <-------------------------------------------->
                               w_sim

    """

    core_width: float
    core_thickness: float
    p_offset: float = 0.0
    n_offset: float = 0.0
    ppp_offset: float = 0.5 * um
    npp_offset: float = 0.5 * um
    slab_thickness: float
    box_thickness: float = 2.0 * um
    clad_thickness: float = 2.0 * um
    p_conc: float = 1e17
    n_conc: float = 1e17
    ppp_conc: float = 1e17
    nnn_conc: float = 1e17
    xmargin: float = 0.5 * um
    xcontact: float = 0.25 * um
    pp_res_x: float = 20 * nm
    pp_p_res_x: float = 5 * nm
    p_res_x: float = 10 * nm
    pn_res_x: float = 1 * nm
    coarse_res_y: float = 10 * nm
    slab_res_y: float = 2 * nm
    atol: float = 1e8
    rtol: float = 1e-8
    max_iter: int = 60

    class Config:
        """Enable adding new."""

        extra = Extra.allow

    # @property
    # def t_sim(self):
    #     return self.box_thickness + self.core_thickness + self.clad_thickness

    # @property
    # def w_sim(self):
    #     return 2 * self.xmargin + self.ppp_offset + self.npp_offset

    def create_2d_mesh(self, device) -> None:
        """Creates a 2D mesh."""
        xmin = (-self.xmargin - self.ppp_offset - self.core_width / 2) / cm
        xmax = (self.xmargin + self.npp_offset + self.core_width / 2) / cm
        self.xppp = -self.ppp_offset - self.core_width / 2
        self.xnpp = self.npp_offset + self.core_width / 2
        ymin = 0 / cm
        ymax = (self.core_thickness) / cm
        xmin_waveguide = (-self.core_width / 2) / cm
        xmax_waveguide = (self.core_width / 2) / cm
        yslab = (self.slab_thickness) / cm

        devsim.create_2d_mesh(mesh="dio")
        devsim.add_2d_mesh_line(mesh="dio", dir="x", pos=xmin, ps=self.pp_res_x / cm)
        devsim.add_2d_mesh_line(
            mesh="dio", dir="x", pos=self.xppp / cm, ps=self.pp_p_res_x / cm
        )
        devsim.add_2d_mesh_line(
            mesh="dio", dir="x", pos=self.xppp / cm / 2, ps=self.p_res_x / cm
        )
        devsim.add_2d_mesh_line(
            mesh="dio", dir="x", pos=-self.p_offset / cm, ps=self.pn_res_x / cm
        )
        devsim.add_2d_mesh_line(
            mesh="dio", dir="x", pos=self.n_offset / cm, ps=self.pn_res_x / cm
        )
        devsim.add_2d_mesh_line(
            mesh="dio", dir="x", pos=self.xnpp / cm / 2, ps=self.p_res_x / cm
        )
        devsim.add_2d_mesh_line(
            mesh="dio", dir="x", pos=self.xnpp / cm, ps=self.pp_p_res_x / cm
        )
        devsim.add_2d_mesh_line(mesh="dio", dir="x", pos=xmax, ps=self.pp_res_x / cm)
        devsim.add_2d_mesh_line(
            mesh="dio", dir="y", pos=ymin, ps=self.coarse_res_y / cm
        )
        devsim.add_2d_mesh_line(mesh="dio", dir="y", pos=yslab, ps=self.slab_res_y / cm)
        devsim.add_2d_mesh_line(
            mesh="dio", dir="y", pos=ymax, ps=self.coarse_res_y / cm
        )

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
            xh=xmin + self.xcontact / cm,
            yl=yslab,
            yh=yslab + 5 * nm / cm,
        )
        devsim.add_2d_region(
            mesh="dio",
            material="Si",
            region="right_contact",
            xl=xmax - self.xcontact / cm,
            xh=xmax,
            yl=yslab,
            yh=yslab + 5 * nm / cm,
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
            xh=xmin + self.xmargin / cm / 2,
        )
        devsim.add_2d_contact(
            mesh="dio",
            name="right",
            material="metal",
            region="slab",
            yl=ymin,
            yh=yslab,
            xl=xmax,
            xh=xmax - self.xmargin / cm / 2,
        )
        devsim.finalize_mesh(mesh="dio")
        devsim.create_device(mesh="dio", device=device)
        self.save_device("test.dat")

    def set_parameters(self, device) -> None:
        """Set parameters for 300 K."""
        simple_physics.SetSiliconParameters(device, "slab", 300)
        simple_physics.SetSiliconParameters(device, "core", 300)
        simple_physics.SetSiliconParameters(device, "left_contact", 300)
        simple_physics.SetSiliconParameters(device, "right_contact", 300)

    def set_net_doping(self, device) -> None:
        """NetDoping."""
        model_create.CreateNodeModel(
            device,
            "slab",
            "Acceptors",
            f"{self.p_conc:1.3e}*step({-1*self.p_offset / cm:1.3e}-x) + {self.ppp_conc:1.3e}*step({self.xppp / cm:1.3e}-x)",
        )
        model_create.CreateNodeModel(
            device,
            "slab",
            "Donors",
            f"{self.n_conc:1.3e}*step(x-{self.n_offset / cm:1.3e}) + {self.nnn_conc:1.3e}*step(x-{self.xnpp / cm:1.3e})",
        )
        model_create.CreateNodeModel(device, "slab", "NetDoping", "Donors-Acceptors")
        model_create.CreateNodeModel(
            device,
            "core",
            "Acceptors",
            f"{self.p_conc:1.1e}*step({-1*self.p_offset / cm:1.6e}-x)",
        )
        model_create.CreateNodeModel(
            device,
            "core",
            "Donors",
            f"{self.n_conc:1.1e}*step(x-{self.n_offset / cm:1.6e})",
        )
        model_create.CreateNodeModel(device, "core", "NetDoping", "Donors-Acceptors")
        model_create.CreateNodeModel(device, "left_contact", "NetDoping", "0")
        model_create.CreateNodeModel(device, "right_contact", "NetDoping", "0")

        self.save_device("test_doping.dat")

    def initial_solution(self, device, region, circuit_contacts=None) -> None:
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

    def drift_diffusion_initial_solution(
        self, device, region, circuit_contacts=None
    ) -> None:
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

    def ddsolver(self) -> None:
        """Initialize mesh and solver."""
        device = "MyDevice"
        self.create_2d_mesh(device)
        self.set_parameters(device=device)
        self.set_net_doping(device=device)

        model_create.CreateSolution(device, "core", "Potential")
        simple_physics.CreateSiliconPotentialOnly(device, "core")
        self.initial_solution(device=device, region="slab")
        self.initial_solution(device=device, region="right_contact")
        self.initial_solution(device=device, region="left_contact")

        # model_create.CreateSolution(device, "left_clad", "Potential")
        # CreateOxidePotentialOnly(device, "left_clad")
        # self.initial_solution(device=device, region="left_clad")

        # model_create.CreateSolution(device, "right_clad", "Potential")
        # CreateOxidePotentialOnly(device, "right_clad")
        # self.initial_solution(device=device, region="right_clad")

        for region in devsim.get_region_list(device=device):
            self.drift_diffusion_initial_solution(device=device, region=region)
        for interface in devsim.get_interface_list(device=device):
            simple_physics.CreateSiliconSiliconInterface(
                device=device, interface=interface
            )

        devsim.solve(
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
            devsim.set_parameter(
                device=device, name=simple_physics.GetContactBiasName("left"), value=V
            )
            devsim.solve(
                type="dc",
                absolute_error=self.atol,
                relative_error=self.rtol,
                maximum_iterations=self.max_iter,
            )
            V += Vstep

    def get_field(self, region_name="core", field_name="Electrons"):
        device = "MyDevice"
        return devsim.get_node_model_values(
            device=device, region=region_name, name=field_name
        )

    def save_device(self, filepath) -> None:
        """Save Device to a tecplot filepath that you can open with Paraview."""
        devsim.write_devices(file=filepath, type="tecplot")

    def plot(
        self,
        tempfile: str = "temp.dat",
        scalars: Optional[str] = None,
        log_scale: bool = False,
        cmap: str = "RdBu",
        jupyter_backend: str = "None",
    ) -> None:
        """Shows the geometry.

        Args:
            tempfile: tempfile path.
            scalars: optional string to plot a field as color over the mesh.
                For instance, acceptor concentration and donor concentration for the PN junction.
            logscale: plots in logscale.
            cmap: color map.
            jupyter_backend: for the jupyter notebook.
        """
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
        disable_print()
        _ = plotter.show(jupyter_backend=jupyter_backend)
        enable_print()

    def list_fields(self, tempfile="temp.dat"):
        """Returns the header of the mesh, which lists all possible fields."""
        devsim.write_devices(file=tempfile, type="tecplot")
        reader = pv.get_reader(tempfile)
        mesh = reader.read()
        return mesh[0]

    def make_waveguide(
        self,
        wavelength: float,
        box_thickness: float = 2.0,
        clad_thickness: float = 2.0,
        grid_resolution: int = 200,
        perturb: bool = True,
        nmodes: int = 4,
        bend_radius: Optional[float] = None,
        cache: bool = False,
        precision: Precision = "double",
        core_material: MaterialSpec = "si",
        clad_material: MaterialSpec = "sio2",
    ) -> Waveguide:
        """Converts the FEM model to a Waveguide object.

        - Rescales lengths to um
        - Adjusts origin to match Waveguide object convention

        Args:
            wavelength: (um).
            box_thickness: thickness BOX (um).
            clad_thickness: thickness cladding (um).
            xmargin: margin from waveguide edge to each side (um).
            grid_resolution: wavelength resolution of the computation grid.
            perturb: add perturbation.
            nmodes: number of modes to compute.
            bend_radius: optional bend radius (um).
            cache: True uses file cache from PDK.modes_path. False skips cache.
            precision: single or double.

        Returns:
            A tidy3d Waveguide object.

        """
        if perturb:
            # Create temporary Waveguide to get the X, Y, Z positions
            temp_Waveguide = Waveguide(
                wavelength=wavelength,
                core_width=self.core_width / um,
                core_thickness=self.core_thickness / um,
                slab_thickness=self.slab_thickness / um,
                box_thickness=box_thickness,
                clad_thickness=clad_thickness,
                side_margin=max(
                    self.ppp_offset + self.xmargin, self.npp_offset + self.xmargin
                )
                / um,
                grid_resolution=grid_resolution,
                precision=precision,
                core_material=core_material,
                clad_material=clad_material,
                cache=False,
            )

            # Create index perturbation
            x_fem = []
            y_fem = []
            dN_fem = []
            dP_fem = []

            mat_dtype = np.float64 if precision == "double" else np.float32

            for region_name in ["core", "slab"]:
                x_fem.append(
                    np.array(self.get_field(region_name=region_name, field_name="x"))
                )
                y_fem.append(
                    np.array(self.get_field(region_name=region_name, field_name="y"))
                )
                dN_fem.append(
                    np.array(
                        self.get_field(region_name=region_name, field_name="Electrons"),
                        dtype=mat_dtype,
                    )
                )
                dP_fem.append(
                    np.array(
                        self.get_field(region_name=region_name, field_name="Holes"),
                        dtype=mat_dtype,
                    )
                )

            x_fem = np.concatenate(x_fem)
            y_fem = np.concatenate(y_fem)
            dN_fem = np.concatenate(dN_fem)
            dP_fem = np.concatenate(dP_fem)

            dn_fem = dn_carriers(wavelength, dN_fem, dP_fem)
            dk_fem = alpha_to_k(dalpha_carriers(wavelength, dN_fem, dP_fem), wavelength)

            # Interpolate the index perturbation onto the Waveguide grid
            temp_grid = temp_Waveguide.waveguide.mode_solver.simulation.grid
            X = temp_grid.centers.x
            Y = temp_grid.centers.y
            Z = temp_grid.centers.z

            freq0 = td.C_0 / wavelength
            freqs = [freq0]

            x_mesh, y_mesh, z_mesh, freq_mesh = np.meshgrid(
                X, Y, Z, freqs, indexing="ij"
            )
            x_mesh_2D = x_mesh[:, :, 0, 0]
            y_mesh_2D = y_mesh[:, :, 0, 0]

            dn_grid = np.nan_to_num(
                griddata(
                    (x_fem * cm / um, y_fem * cm / um),
                    dn_fem,
                    (x_mesh_2D, y_mesh_2D),
                    DEVSIM_INTERPOLATION_METHOD,
                )
            )
            dk_grid = np.nan_to_num(
                griddata(
                    (x_fem * cm / um, y_fem * cm / um),
                    dk_fem,
                    (x_mesh_2D, y_mesh_2D),
                    DEVSIM_INTERPOLATION_METHOD,
                )
            )

            # Extend to 3D (Nx, Ny,) -> (Nx, Ny, Nz,)
            dn_grid_3D = np.stack([dn_grid] * Z.size, axis=-1)
            dk_grid_3D = np.stack([dk_grid] * Z.size, axis=-1)

            # Extend to 4D (Nx, Ny, Nz,) -> (Nx, Ny, Nz, Nf)
            dn_grid_td = np.stack([dn_grid_3D] * len(freqs), axis=-1)
            dk_grid_td = np.stack([dk_grid_3D] * len(freqs), axis=-1)

            n_core, k_core = get_nk(core_material)
            n_dataset = td.ScalarFieldDataArray(
                n_core + dn_grid_td, coords=dict(x=X, y=Y, z=Z, f=freqs)
            )
            k_dataset = td.ScalarFieldDataArray(
                k_core + dk_grid_td, coords=dict(x=X, y=Y, z=Z, f=freqs)
            )

            # core_material_pertub = td.CustomMedium.from_nk(
            #     n=n_dataset,
            #     k=k_dataset,
            #     interp_method="linear",
            # )
            if isinstance(core_material, str):
                name_material = core_material + "_with_carriers"
            else:
                name_material = "core_material_with_carriers"

            eps_real = n_dataset**2 - k_dataset**2
            eps_imag = 2 * n_dataset * k_dataset
            eps_diag_data = td.ScalarFieldDataArray(
                eps_real + 1.0j * eps_imag, coords=dict(x=X, y=Y, z=Z, f=freqs)
            )

            eps_perturb = td.PermittivityDataset(
                eps_xx=eps_diag_data, eps_yy=eps_diag_data, eps_zz=eps_diag_data
            )
            core_material_pertub = td.CustomMedium(
                name=name_material,
                eps_dataset=eps_perturb,
                frequency_range=[0.99 * freq0, 1.01 * freq0],
            )

            # dn_dict = (
            #     {
            #         "x": x_fem * cm / um,
            #         "y": y_fem * cm / um + box_thickness,
            #         "dn": dn_fem + 1j * dk_fem,
            #     }
            #     if perturb
            #     else None
            # )

            # Create perturbed waveguide, handle like regular mode
            return Waveguide(
                wavelength=wavelength,
                core_width=self.core_width / um,
                core_thickness=self.core_thickness / um,
                slab_thickness=self.slab_thickness / um,
                box_thickness=box_thickness,
                clad_thickness=clad_thickness,
                side_margin=max(
                    self.ppp_offset + self.xmargin, self.npp_offset + self.xmargin
                )
                / um,
                grid_resolution=grid_resolution,
                precision=precision,
                core_material=core_material_pertub,
                clad_material=clad_material,
                cache=cache,
            )
        else:
            # Create simple waveguide, handle like regular mode
            return Waveguide(
                wavelength=wavelength,
                core_width=self.core_width / um,
                core_thickness=self.core_thickness / um,
                slab_thickness=self.slab_thickness / um,
                box_thickness=box_thickness,
                clad_thickness=clad_thickness,
                side_margin=max(
                    self.ppp_offset + self.xmargin, self.npp_offset + self.xmargin
                )
                / um,
                grid_resolution=grid_resolution,
                precision=precision,
                core_material=core_material,
                clad_material=clad_material,
                cache=cache,
            )


def clear_devsim_cache() -> None:
    try:
        devsim.delete_mesh(mesh="dio")
    except devsim.error:
        warnings.warn("No mesh to delete.")

    try:
        devsim.delete_device(device="MyDevice")
    except devsim.error:
        warnings.warn("No device to delete.")


if __name__ == "__main__":
    c = PINWaveguide(
        core_width=500 * nm,
        core_thickness=220 * nm,
        slab_thickness=90 * nm,
    )
    c.ddsolver()
    c.save_device("test.dat")

    import os
    import shutil

    foldername = "04_wabsorption"
    if os.path.exists(foldername) and os.path.isdir(foldername):
        shutil.rmtree(foldername)
    os.mkdir(foldername)

    voltage_solver_step = -0.2
    voltages = np.arange(0, -1, voltage_solver_step)
    # voltages = [-0.1]
    # voltages = [0]

    neffs_doped = []
    indices_doped = []

    c_control = c.make_waveguide(wavelength=1.55, perturb=False, precision="double")
    c_control.compute_modes()
    indices_control = c_control.nx
    neffs_control = c_control.neffs[0]
    for voltage in voltages:
        c.ramp_voltage(voltage, voltage_solver_step)
        c_doped = c.make_waveguide(wavelength=1.55, precision="double")
        c_doped.compute_modes(isolate=True)
        indices_doped.append(c_doped.nx)
        # c2.plot_index()
        neffs_doped.append(c_doped.neffs[0])
        # c2.plot_Ex()
        # c.save_device(f"./{foldername}/test_v_{voltage}.dat")

        plt.figure()
        plt.imshow(
            np.log10(np.abs(c_doped.nx.T - indices_control.T)),
            origin="lower",
            vmin=-14,
            vmax=-3,
        )
        plt.colorbar()
        plt.savefig(f"./{foldername}/indices_test_shift_{voltage}.png")

    plt.figure()
    plt.plot(voltages, np.array(neffs_doped) - neffs_control)
    plt.xlabel("Voltage (V)")
    plt.ylabel("delta neff")
    plt.savefig(f"./{foldername}/neff_test_shift.png")

    plt.figure()
    plt.plot(voltages, np.imag(neffs_doped))
    plt.xlabel("Voltage (V)")
    plt.ylabel("delta abs")
    plt.savefig(f"./{foldername}/neff_test_abs.png")

    # import pickle

    # diffs = []

    # for voltage in [0.0, -0.3, -0.6]:
    #     filepath = f'./02_reverse/test_v_{voltage}.dat'
    #     devsim.load_devices(file=filepath)
    #     electrons = []
    #     for region_name in ["core", "slab"]:
    #         electrons.append(devsim.get_node_model_values(device='MyDevice', region=region_name, name="Electrons"))
    #         electrons.append(devsim.get_node_model_values(device='MyDevice', region=region_name, name="Electrons"))

    #     filepath = f'./02_reverse/test_v_{voltage}_electrons.dat'

    #     dbfile = open(filepath, 'ab')
    #     pickle.dump(electrons, dbfile)
    #     dbfile.close()
