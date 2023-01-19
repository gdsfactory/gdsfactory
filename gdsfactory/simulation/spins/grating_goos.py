"""2D grating coupler optimization"""


import os
import pickle

from pydantic import BaseModel
import matplotlib.pyplot as plt
import numpy as np

from spins import goos
from spins.goos_sim import maxwell


class Options(BaseModel):
    """Maintains list of options for the optimization.

    Attributes:
        coupler_len: Length of the grating coupler.
        wg_width: Width of the grating coupler. Only relevant for GDS file generation.
        wg_len: Length of the waveguide to which the grating coupler couples.
        wg_thickness: Thickness of the waveguide.
        etch_frac: Etch fraction of the grating.
            (wg_thickness-slab_thickness)/wg_thickness
        min_features: Minimum feature sizes.
        box_size: Thickness of the buried oxide layer.
        source_angle_deg: Angle of the Gaussian beam in degrees relative to the normal.
        buffer_len: Additional distance to add to the top and bottom of the simulation.

        eps_bg: Refractive index of the background.
        eps_fg: Refraction index of the waveguide/grating.

        beam_dist: Distance of the Gaussian beam from the grating.
        beam_width: Diameter of the Gaussian beam.
        beam_extents: Length of the Gaussian beam to use in the simulation.

        wavelength: Wavelength to simulate at.
        dx: Grid spacing to use in the simulation.
        pixel_size: Pixel size of the continuous grating coupler
            parametrization.
        cont_max_iter: Number of iterations to run in continuous optimization.
    """

    coupler_len: float = 12000
    wg_width: float = 10000
    wg_len: float = 2400
    wg_thickness: float = 220
    box_size: float = 2000
    source_angle_deg: float = -10
    buffer_len: float = 2000
    eps_bg: float = 1.444
    eps_wg: float = 3.4765
    beam_dist: float = 1000
    beam_width: float = 10400
    beam_extents: float = 14000
    wavelength: float = 1550
    dx: float = 20
    pixel_size: float = 20
    etch_frac: float = 0.5
    min_features: float = 100
    cont_max_iter: int = 20


def run_opt(out_folder_name: str = "grating_full_opt", **settings) -> None:
    """Run grating coupler optimization.

    Args:
        out_folder_name: folder name to store optimization.

    Keyword Args:
        coupler_len: Length of the grating coupler.
        wg_width: Width of the grating coupler. Only relevant for GDS file generation.
        wg_len: Length of the waveguide to which the grating coupler couples.
        wg_thickness: Thickness of the waveguide.
        etch_frac: Etch fraction of the grating.
            (wg_thickness-slab_thickness)/wg_thickness
        min_features: Minimum feature sizes.
        box_size: Thickness of the buried oxide layer.
        source_angle_deg: Angle of the Gaussian beam in degrees relative to the normal.
        buffer_len: Additional distance to add to the top and bottom of the simulation.

        eps_bg: Refractive index of the background.
        eps_fg: Refraction index of the waveguide/grating.

        beam_dist: Distance of the Gaussian beam from the grating.
        beam_width: Diameter of the Gaussian beam.
        beam_extents: Length of the Gaussian beam to use in the simulation.

        wavelength: Wavelength to simulate at.
        dx: Grid spacing to use in the simulation.
        pixel_size: Pixel size of the continuous grating coupler
            parametrization.
        cont_max_iter: Number of iterations to run in continuous optimization.

    """
    params = Options(**settings)
    # set-up with saving folder, and optimization plan
    folder_plt = out_folder_name  # Plotting folder is separately here, in case one wishes to plot from another folder.
    out_folder = os.path.join(os.getcwd(), out_folder_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    goos.util.setup_logging(out_folder)
    plan = goos.OptimizationPlan(save_path=out_folder)
    cont_max_iter = params.cont_max_iter

    # set-up background shapes
    with plan:
        substrate = goos.Cuboid(
            pos=goos.Constant(
                [
                    params.coupler_len / 2,
                    0,
                    -params.box_size - params.wg_thickness / 2 - 5000,
                ]
            ),
            extents=goos.Constant([params.coupler_len + 10000, 1000, 10000]),
            material=goos.material.Material(index=params.eps_wg),
        )

        waveguide = goos.Cuboid(
            pos=goos.Constant([-params.wg_len / 2, 0, 0]),
            extents=goos.Constant(
                [params.wg_len, params.wg_width, params.wg_thickness]
            ),
            material=goos.material.Material(index=params.eps_wg),
        )

        wg_bottom = goos.Cuboid(
            pos=goos.Constant(
                [params.coupler_len / 2, 0, -params.wg_thickness / 2 * params.etch_frac]
            ),
            extents=goos.Constant(
                [
                    params.coupler_len,
                    params.wg_width,
                    params.wg_thickness * (1 - params.etch_frac),
                ]
            ),
            material=goos.material.Material(index=params.eps_wg),
        )

    # set-up design area and finish eps we need.

    with plan:

        def initializer(size):
            return np.random.random(size)

        # Continuous optimization.
        var, design_cont = goos.pixelated_cont_shape(
            initializer=initializer,
            pos=goos.Constant(
                [
                    params.coupler_len / 2,
                    0,
                    params.wg_thickness / 2 * (1 - params.etch_frac),
                ]
            ),
            extents=[
                params.coupler_len,
                params.wg_width,
                params.wg_thickness * params.etch_frac,
            ],
            material=goos.material.Material(index=params.eps_bg),
            material2=goos.material.Material(index=params.eps_wg),
            pixel_size=[params.pixel_size, params.wg_width, params.wg_thickness],
        )

        eps_cont = goos.GroupShape([substrate, waveguide, wg_bottom, design_cont])

    # Set-up continuous optimization objective function with eps.
    with plan:

        sim_left_x = -params.wg_len
        sim_right_x = params.coupler_len + params.buffer_len
        pml_thick = params.dx * 10
        sim_z_center = (
            params.wg_thickness / 2 + params.beam_dist - params.box_size
        ) / 2
        sim_z_extent = (
            params.wg_thickness
            + params.beam_dist
            + params.box_size
            + 2000
            + pml_thick * 2
        )

        sources = [
            maxwell.GaussianSource(
                w0=params.beam_width / 2,
                center=[
                    params.coupler_len / 2,
                    0,
                    params.wg_thickness / 2 + params.beam_dist,
                ],
                extents=[params.beam_extents, 0, 0],
                normal=[0, 0, -1],
                power=1,
                theta=np.deg2rad(params.source_angle_deg),
                psi=np.pi / 2,
                polarization_angle=0,
                normalize_by_sim=True,
            )
        ]
        outputs = [
            maxwell.Epsilon(name="eps"),
            maxwell.ElectricField(name="field"),
            maxwell.WaveguideModeOverlap(
                name="overlap",
                center=[-params.wg_len / 2, 0, 0],
                extents=[0, 1000, 2000],
                normal=[-1, 0, 0],
                mode_num=0,
                power=1,
            ),
        ]
        simulation_space = maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=params.dx),
            sim_region=goos.Box3d(
                center=[(sim_left_x + sim_right_x) / 2, 0, sim_z_center],
                extents=[sim_right_x - sim_left_x, 0, sim_z_extent],
            ),
            pml_thickness=[pml_thick, pml_thick, 0, 0, pml_thick, pml_thick],
        )

        sim_cont = maxwell.fdfd_simulation(
            name="sim_cont",
            simulation_space=simulation_space,
            wavelength=params.wavelength,
            sources=sources,
            eps=eps_cont,
            solver="local_direct",
            outputs=outputs,
            background=goos.material.Material(index=1.444),
        )

    obj_c = (
        1 - goos.abs(sim_cont["overlap"])
    ) ** 2  # elaborate how simple. It makes difference. This from our experience is the best. Try your options!
    obj_c = goos.rename(obj_c, name="obj_cont")

    # set-up continuous optimization with scipy
    with plan:
        goos.opt.scipy_minimize(
            obj_c,
            "L-BFGS-B",
            monitor_list=[
                sim_cont["eps"],
                sim_cont["field"],
                sim_cont["overlap"],
                obj_c,
            ],
            max_iters=cont_max_iter,
            name="opt_cont",
        )

        # Prevent optimization from optimizing over continuous variable.
        var.freeze()

    # set-up discretization.
    with plan:
        (
            grating_var,
            height_var,
            design_disc,
        ) = goos.grating.discretize_to_pixelated_grating(
            var,
            height_fracs=[0, 1],
            pixel_size=params.pixel_size,
            start_height_ind=1,
            end_height_ind=1,
            min_features=params.min_features,
            pos=[
                params.coupler_len / 2,
                0,
                params.wg_thickness / 2 * (1 - params.etch_frac),
            ],
            extents=[
                params.coupler_len,
                params.wg_width,
                params.wg_thickness * params.etch_frac,
            ],
            material=goos.material.Material(index=params.eps_bg),
            material2=goos.material.Material(index=params.eps_wg),
            grating_dir=0,
            grating_dir_spacing=20,
            etch_dir=2,
            etch_dir_divs=1,
        )
        eps_disc = goos.GroupShape([substrate, waveguide, wg_bottom, design_disc])

    # Set-up discrete optimization objective function with eps.
    with plan:

        sim_left_x = -params.wg_len
        sim_right_x = params.coupler_len + params.buffer_len
        pml_thick = params.dx * 10
        sim_z_center = (
            params.wg_thickness / 2 + params.beam_dist - params.box_size
        ) / 2
        sim_z_extent = (
            params.wg_thickness
            + params.beam_dist
            + params.box_size
            + 2000
            + pml_thick * 2
        )

        simulation_space = maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=params.dx),
            sim_region=goos.Box3d(
                center=[(sim_left_x + sim_right_x) / 2, 0, sim_z_center],
                extents=[sim_right_x - sim_left_x, 0, sim_z_extent],
            ),
            pml_thickness=[pml_thick, pml_thick, 0, 0, pml_thick, pml_thick],
        )

        sources = [
            maxwell.GaussianSource(
                w0=params.beam_width / 2,
                center=[
                    params.coupler_len / 2,
                    0,
                    params.wg_thickness / 2 + params.beam_dist,
                ],
                extents=[params.beam_extents, 0, 0],
                normal=[0, 0, -1],
                power=1,
                theta=np.deg2rad(params.source_angle_deg),
                psi=np.pi / 2,
                polarization_angle=0,
                normalize_by_sim=True,
            )
        ]

        outputs = [
            maxwell.Epsilon(name="eps"),
            maxwell.ElectricField(name="field"),
            maxwell.WaveguideModeOverlap(
                name="overlap",
                center=[-params.wg_len / 2, 0, 0],
                extents=[0, 1000, 2000],
                normal=[-1, 0, 0],
                mode_num=0,
                power=1,
            ),
        ]
        sim_disc = maxwell.fdfd_simulation(
            name="sim_disc",
            simulation_space=simulation_space,
            wavelength=params.wavelength,
            sources=sources,
            eps=eps_disc,
            solver="local_direct",
            outputs=outputs,
            background=goos.material.Material(index=params.eps_bg),
        )

    obj_d = (1 - goos.abs(sim_disc["overlap"])) ** 2  # elaborate how simple
    obj_d = goos.rename(obj_d, name="obj_disc")

    # set-up discrete optimization with scipy
    with plan:
        goos.opt.scipy_minimize(
            obj_d,
            "L-BFGS-B",
            monitor_list=[
                sim_disc["eps"],
                sim_disc["field"],
                sim_disc["overlap"],
                obj_d,
            ],
            max_iters=20,
            name="opt_disc",
            ftol=1e-8,
        )

    # run the optimization
    with plan:
        plan.save()
        plan.run()

    # visualizing the initial structure permittivity and the field.
    with open(os.path.join(folder_plt, "step1.pkl"), "rb") as fp:
        data = pickle.load(fp)

        plt.figure(figsize=(10, 12))
        plt.imshow(
            np.rot90(
                np.abs(data["monitor_data"]["sim_cont.eps"][0].squeeze()), 1, (0, 1)
            )
        )
        plt.axis("off")
        plt.tight_layout()
        plt.figure(figsize=(10, 12))
        plt.imshow(
            np.rot90(
                np.abs(data["monitor_data"]["sim_cont.field"][1].squeeze()), 1, (0, 1)
            )
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        print(
            "Overlap transmission value is "
            + str(np.abs(data["monitor_data"]["sim_cont.overlap"]) ** 2)
        )

    # visualizing end of continuous optimization
    with open(os.path.join(folder_plt, f"step{cont_max_iter}.pkl"), "rb") as fp:
        data = pickle.load(fp)

        plt.figure(figsize=(10, 12))
        plt.imshow(
            np.rot90(
                np.abs(data["monitor_data"]["sim_cont.eps"][0].squeeze()), 1, (0, 1)
            )
        )
        plt.axis("off")
        plt.tight_layout()
        plt.figure(figsize=(10, 12))
        plt.imshow(
            np.rot90(
                np.abs(data["monitor_data"]["sim_cont.field"][1].squeeze()), 1, (0, 1)
            )
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        print(
            "Overlap transmission value is "
            + str(np.abs(data["monitor_data"]["sim_cont.overlap"]) ** 2)
        )

    # visualizing the structure and the field at the end of the discretization
    with open(os.path.join(folder_plt, f"step{cont_max_iter + 1}.pkl"), "rb") as fp:
        data = pickle.load(fp)

        plt.figure(figsize=(10, 12))
        plt.imshow(
            np.rot90(
                np.abs(data["monitor_data"]["sim_disc.eps"][0].squeeze()), 1, (0, 1)
            )
        )
        plt.axis("off")
        plt.tight_layout()
        plt.figure(figsize=(10, 12))
        plt.imshow(
            np.rot90(
                np.abs(data["monitor_data"]["sim_disc.field"][1].squeeze()), 1, (0, 1)
            )
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        print(
            "Overlap transmission value is "
            + str(np.abs(data["monitor_data"]["sim_disc.overlap"]) ** 2)
        )

    # visualizing the structure and the field at the end of the optimization
    step = goos.util.get_latest_log_step(folder_plt)
    with open(os.path.join(folder_plt, f"step{step}.pkl"), "rb") as fp:
        data = pickle.load(fp)

    plt.figure(figsize=(10, 12))
    plt.imshow(
        np.rot90(np.abs(data["monitor_data"]["sim_disc.eps"][0].squeeze()), 1, (0, 1))
    )
    plt.axis("off")
    plt.tight_layout()

    plt.figure(figsize=(10, 12))
    plt.imshow(
        np.rot90(np.abs(data["monitor_data"]["sim_disc.field"][1].squeeze()), 1, (0, 1))
    )
    plt.axis("off")
    plt.tight_layout()

    plt.show()
    print(
        "Overlap transmission value is "
        + str(np.abs(data["monitor_data"]["sim_disc.overlap"]) ** 2)
    )

    # Reading all pkl files in the saving folder to see optimization trajectory over iterations.
    disc_last_step = goos.util.get_latest_log_step(folder_plt)
    transmission = []
    for step in range(1, cont_max_iter + 1):
        with open(os.path.join(folder_plt, f"step{step}.pkl"), "rb") as fp:
            data = pickle.load(fp)
            transmission.append(np.abs(data["monitor_data"]["sim_cont.overlap"]) ** 2)
    for step in range(cont_max_iter + 1, int(disc_last_step) + 1):
        with open(os.path.join(folder_plt, f"step{step}.pkl"), "rb") as fp:
            data = pickle.load(fp)
            transmission.append(np.abs(data["monitor_data"]["sim_disc.overlap"]) ** 2)

    # plotting the overlap values for the all pkl files in the saving folder to see optimization trajectory over iterations.
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, int(disc_last_step) + 1), transmission)
    plt.xlabel("Iteration")
    plt.ylabel("Transmission")
    plt.tight_layout()
