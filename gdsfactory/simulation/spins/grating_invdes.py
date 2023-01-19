"""2D fiber-to-chip grating coupler optimization code.

This is a simple spins example that optimizes a fiber-to-chip grating coupler
for the SOI platform. See Su et al. Optics Express (2018) for details.

To run an optimization:
$ python3 grating.py run save-folder

To view results:
$ python3 grating.py view save-folder

To see optimization status quickly:
$ python3 grating.py view_quick save-folder

To resume an optimization:
$ python3 grating.py resume save-folder

To generate a GDS file of the grating:
$ python3 grating.py gen_gds save-folder

Deprecated, use grating_goos instead which uses the new library
"""
import os
import pickle
import shutil
import pathlib
from typing import List, Tuple

import gdspy
import numpy as np

from spins.invdes import problem_graph
from spins.invdes.problem_graph import log_tools
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace

# If `True`, also minimize the back-reflection.
MINIMIZE_BACKREFLECTION = False
# If 'True`, runs an additional `cont_iters' of continuous optimization with
# discreteness permittivity biasing penalty added.
# Fine-tuning the `intial_value' of `disc_scaling may be necessary depending
# on application and the number of wavelengths optimized.
DISCRETENESS_PENALTY = True


def run_opt(
    save_folder: str,
    grating_len: float = 15e3,
    wg_width: float = 10e3,
    wg_thickness: float = 220,
    etch_frac: float = 0.5,
    box_thickness: float = 2000,
    dx: int = 40,
    num_pmls: int = 10,
    cont_iters: int = 50,
    disc_iters: int = 200,
    min_feature: float = 80,
    buffer_len: float = 1500,
    visualize: bool = True,
) -> None:
    """Run main optimization script.

    This function setups the optimization and executes it.

    Args:
        save_folder: Location to save the optimization data.
        grating_len: Length of the grating coupler and design region.
        wg_width: Width of the waveguide.
        wg_thickness: Thickness of the waveguide.
        etch_frac: Etch fraction of the grating. 1.0 indicates a fully-etched grating.
        box_thickness: Thickness of BOX layer in nm.
        dx: Grid spacing to use.
        num_pmls: Number of PML layers to use on each side.
        cont_iters: Number of iterations to run in continuous optimization.
        disc_iters: Number of iterations to run in discrete optimization.
        min_feature: Minimum feature size in nanometers.
        buffer_len: Buffer distance to put between grating and the end of the
            simulation region. This excludes PMLs.
    """
    os.makedirs(save_folder)

    sim_space = create_sim_space(
        "sim_fg.gds",
        "sim_bg.gds",
        grating_len=grating_len,
        box_thickness=box_thickness,
        wg_thickness=wg_thickness,
        etch_frac=etch_frac,
        wg_width=wg_width,
        dx=dx,
        num_pmls=num_pmls,
        buffer_len=buffer_len,
        visualize=visualize,
    )
    obj, monitors = create_objective(
        sim_space, wg_thickness=wg_thickness, grating_len=grating_len
    )
    trans_list = create_transformations(
        obj,
        monitors,
        cont_iters=cont_iters,
        disc_iters=disc_iters,
        sim_space=sim_space,
        min_feature=min_feature,
    )
    plan = optplan.OptimizationPlan(transformations=trans_list)

    # Save the optimization plan so we have an exact record of all the
    # parameters.
    with open(os.path.join(save_folder, "optplan.json"), "w") as fp:
        fp.write(optplan.dumps(plan))
    # Copy over the GDS files.
    shutil.copyfile("sim_fg.gds", os.path.join(save_folder, "sim_fg.gds"))
    shutil.copyfile("sim_bg.gds", os.path.join(save_folder, "sim_bg.gds"))

    # Execute the optimization and indicate that the current folder (".") is
    # the project folder. The project folder is the root folder for any
    # auxiliary files (e.g. GDS files).
    problem_graph.run_plan(plan, ".", save_folder=save_folder)

    # Generate the GDS file.
    gen_gds(save_folder, grating_len, wg_width)


def create_sim_space(
    gds_fg_name: str,
    gds_bg_name: str,
    grating_len: float = 12000,
    etch_frac: float = 0.5,
    box_thickness: float = 2000,
    wg_width: float = 12000,
    wg_thickness: float = 220,
    buffer_len: float = 1500,
    dx: int = 40,
    num_pmls: int = 10,
    visualize: bool = True,
) -> optplan.SimulationSpace:
    """Creates the simulation space.

    The simulation space contains information about the boundary conditions,
    gridding, and design region of the simulation.

    Args:
        gds_fg_name: Location to save foreground GDS.
        gds_bg_name: Location to save background GDS.
        grating_len: Length of the grating coupler and design region.
        etch_frac: Etch fraction of the grating. 1.0 indicates a fully-etched
            grating.
        box_thickness: Thickness of BOX layer in nm.
        wg_thickness: Thickness of the waveguide.
        wg_width: Width of the waveguide.
        buffer_len: Buffer distance to put between grating and the end of the
            simulation region. This excludes PMLs.
        dx: Grid spacing to use.
        num_pmls: Number of PML layers to use on each side.
        visualize: if True plots fields.

    Returns:
        A `SimulationSpace` description.
    """
    # Calculate the simulation size, including  PMLs
    sim_size = [
        grating_len + 2 * buffer_len + dx * num_pmls,
        wg_width + 2 * buffer_len + dx * num_pmls,
    ]
    # First, we use `gdspy` to draw the waveguides and shapes that we would
    # like to use. Instead of programmatically generating a GDS file using
    # `gdspy`, we could also simply provide a GDS file (e.g. drawn using
    # KLayout).

    # Declare some constants to represent the different layers.
    LAYER_SILICON_ETCHED = 100
    LAYER_SILICON_NONETCHED = 101

    # Create rectangles corresponding to the waveguide, the BOX layer, and the
    # design region. We extend the rectangles outside the simulation region
    # by multiplying locations by a factor of 1.1.

    # We distinguish between the top part of the waveguide (which is etched)
    # and the bottom part of the waveguide (which is not etched).
    waveguide_top = gdspy.Rectangle(
        (-1.1 * sim_size[0] / 2, -wg_width / 2),
        (-grating_len / 2, wg_width / 2),
        LAYER_SILICON_ETCHED,
    )
    waveguide_bottom = gdspy.Rectangle(
        (-1.1 * sim_size[0] / 2, -wg_width / 2),
        (grating_len / 2, wg_width / 2),
        LAYER_SILICON_NONETCHED,
    )
    design_region = gdspy.Rectangle(
        (-grating_len / 2, -wg_width / 2),
        (grating_len / 2, wg_width / 2),
        LAYER_SILICON_ETCHED,
    )

    # Generate the foreground and background GDS files.
    gds_fg = gdspy.Cell("FOREGROUND", exclude_from_current=True)
    gds_fg.add(waveguide_top)
    gds_fg.add(waveguide_bottom)
    gds_fg.add(design_region)

    gds_bg = gdspy.Cell("BACKGROUND", exclude_from_current=True)
    gds_bg.add(waveguide_top)
    gds_bg.add(waveguide_bottom)

    gdspy.write_gds(gds_fg_name, [gds_fg], unit=1e-9, precision=1e-9)
    gdspy.write_gds(gds_bg_name, [gds_bg], unit=1e-9, precision=1e-9)

    # The BOX layer/silicon device interface is set at `z = 0`.
    #
    # Describe materials in each layer.
    # We actually have four material layers:
    # 1) Silicon substrate
    # 2) Silicon oxide BOX layer
    # 3) Bottom part of grating that is not etched
    # 4) Top part of grating that can be etched.
    #
    # The last two layers put together properly describe a partial etch.
    #
    # Note that the layer numbering in the GDS file is arbitrary. In our case,
    # layer 100 and 101 correspond to actual structure. Layer 300 is a dummy
    # layer; it is used for layers that only have one material (i.e. the
    # background and foreground indices are identical) so the actual structure
    # used does not matter.
    stack = [
        optplan.GdsMaterialStackLayer(
            foreground=optplan.Material(mat_name="Si"),
            background=optplan.Material(mat_name="Si"),
            # Note that layer number here does not actually matter because
            # the foreground and background are the same material.
            gds_layer=[300, 0],
            extents=[-10000, -box_thickness],
        ),
        optplan.GdsMaterialStackLayer(
            foreground=optplan.Material(mat_name="SiO2"),
            background=optplan.Material(mat_name="SiO2"),
            gds_layer=[300, 0],
            extents=[-box_thickness, 0],
        ),
    ]
    # If `etch-frac` is 1, then we do not need two separate layers.
    if etch_frac != 1:
        stack.append(
            optplan.GdsMaterialStackLayer(
                foreground=optplan.Material(mat_name="Si"),
                background=optplan.Material(mat_name="SiO2"),
                gds_layer=[LAYER_SILICON_NONETCHED, 0],
                extents=[0, wg_thickness * (1 - etch_frac)],
            )
        )
    stack.append(
        optplan.GdsMaterialStackLayer(
            foreground=optplan.Material(mat_name="Si"),
            background=optplan.Material(mat_name="SiO2"),
            gds_layer=[LAYER_SILICON_ETCHED, 0],
            extents=[wg_thickness * (1 - etch_frac), wg_thickness],
        )
    )

    mat_stack = optplan.GdsMaterialStack(
        # Any region of the simulation that is not specified is filled with
        # oxide.
        background=optplan.Material(mat_name="SiO2"),
        stack=stack,
    )

    sim_z_start = -box_thickness - 1000
    sim_z_end = wg_thickness + 1500

    # Create a simulation space for both continuous and discrete optimization.
    simspace = optplan.SimulationSpace(
        name="simspace",
        mesh=optplan.UniformMesh(dx=dx),
        eps_fg=optplan.GdsEps(gds=gds_fg_name, mat_stack=mat_stack),
        eps_bg=optplan.GdsEps(gds=gds_bg_name, mat_stack=mat_stack),
        # Note that we explicitly set the simulation region. Anything
        # in the GDS file outside of the simulation extents will not be drawn.
        sim_region=optplan.Box3d(
            center=[0, 0, (sim_z_start + sim_z_end) / 2],
            extents=[sim_size[0], dx, sim_z_end - sim_z_start],
        ),
        selection_matrix_type="uniform",
        # PMLs are applied on x- and z-axes. No PMLs are applied along y-axis
        # because it is the axis of translational symmetry.
        pml_thickness=[num_pmls, num_pmls, 0, 0, num_pmls, num_pmls],
    )

    if visualize:
        # To visualize permittivity distribution, we actually have to
        # construct the simulation space object.
        import matplotlib.pyplot as plt
        from spins.invdes.problem_graph.simspace import get_fg_and_bg

        context = workspace.Workspace()
        eps_fg, eps_bg = get_fg_and_bg(context.get_object(simspace), wlen=1550)

        def plot(x):
            plt.imshow(np.abs(x)[:, 0, :].T.squeeze(), origin="lower")

        plt.figure()
        plt.subplot(3, 1, 1)
        plot(eps_fg[2])
        plt.title("eps_fg")

        plt.subplot(3, 1, 2)
        plot(eps_bg[2])
        plt.title("eps_bg")

        plt.subplot(3, 1, 3)
        plot(eps_fg[2] - eps_bg[2])
        plt.title("design region")
        plt.show()
    return simspace


def create_objective(
    sim_space: optplan.SimulationSpace,
    wg_thickness: float,
    grating_len: float,
    wavelengths: Tuple[float, ...] = (1550,),
) -> Tuple[optplan.Function, List[optplan.Monitor]]:
    """Creates an objective function.

    The objective function is what is minimized during the optimization.

    Args:
        sim_space: The simulation space description.
        wg_thickness: Thickness of waveguide.
        grating_len: Length of grating.
        wavelengths: to optimize over.

    Returns:
        A tuple `(obj, monitor_list)` where `obj` is an objectivce function that
        tries to maximize the coupling efficiency of the grating coupler and
        `monitor_list` is a list of monitors (values to keep track of during
        the optimization.
    """
    # Keep track of metrics and fields that we want to monitor.
    monitor_list = []
    objectives = []

    # Set wavelengths to optimize over
    for wlen in wavelengths:
        epsilon = optplan.Epsilon(
            simulation_space=sim_space,
            wavelength=wlen,
        )
        # Append to monitor list for each wavelength
        monitor_list.append(
            optplan.FieldMonitor(name=f"mon_eps_{wlen}", function=epsilon)
        )

        # Add a Gaussian source that is angled at 10 degrees.
        sim = optplan.FdfdSimulation(
            source=optplan.GaussianSource(
                polarization_angle=0,
                theta=np.deg2rad(
                    -10
                ),  # theta is around X, it only affects mangnitude of simulation
                psi=0,
                # psi=np.pi / 2, # psi rotates around Z
                center=[0, 0, wg_thickness + 700],
                extents=[14000, 14000, 0],
                normal=[0, 0, -1],
                power=1,
                w0=5200,
                normalize_by_sim=True,
            ),
            solver="local_direct",
            wavelength=wlen,
            simulation_space=sim_space,
            epsilon=epsilon,
        )
        monitor_list.append(
            optplan.FieldMonitor(
                name=f"mon_field_{str(wlen)}",
                function=sim,
                normal=[0, 1, 0],
                center=[0, 0, 0],
            )
        )

        wg_overlap = optplan.WaveguideModeOverlap(
            center=[-grating_len / 2 - 1000, 0, wg_thickness / 2],
            extents=[0.0, 1500, 1500.0],
            mode_num=0,
            normal=[-1.0, 0.0, 0.0],
            power=1.0,
        )
        power = optplan.abs(optplan.Overlap(simulation=sim, overlap=wg_overlap)) ** 2
        monitor_list.append(
            optplan.SimpleMonitor(name=f"mon_power_{str(wlen)}", function=power)
        )

        if not MINIMIZE_BACKREFLECTION:
            # Spins minimizes the objective function, so to make `power` maximized,
            # we minimize `1 - power`.
            obj = 1 - power
        else:
            # TODO: Use a Gaussian overlap to calculate power emitted by grating
            # so we only need one simulation to handle backreflection and
            # transmission.
            refl_sim = optplan.FdfdSimulation(
                source=optplan.WaveguideModeSource(
                    center=wg_overlap.center,
                    extents=wg_overlap.extents,
                    mode_num=0,
                    normal=[1, 0, 0],
                    power=1.0,
                ),
                solver="local_direct",
                wavelength=wlen,
                simulation_space=sim_space,
                epsilon=epsilon,
            )
            refl_power = (
                optplan.abs(optplan.Overlap(simulation=refl_sim, overlap=wg_overlap))
                ** 2
            )
            monitor_list.append(
                optplan.SimpleMonitor(
                    name=f"mon_refl_power_{str(wlen)}", function=refl_power
                )
            )

            # We now have two sub-objectives: Maximize transmission and minimize
            # back-reflection, so we must an objective that defines the appropriate
            # tradeoff between transmission and back-reflection. Here, we choose the
            # simplest objective to do this, but you can use SPINS functions to
            # design more elaborate objectives.
            obj = (1 - power) + 4 * refl_power

        objectives.append(obj)

    obj = sum(objectives)

    return obj, monitor_list


def create_transformations(
    obj: optplan.Function,
    monitors: List[optplan.Monitor],
    cont_iters: int,
    disc_iters: int,
    sim_space: optplan.SimulationSpaceBase,
    min_feature: float = 100,
    cont_to_disc_factor: float = 1.1,
) -> List[optplan.Transformation]:
    """Creates a list of transformations for the optimization.

    The grating coupler optimization proceeds as follows:
    1) Continuous optimization whereby each pixel can vary between device and
       background permittivity.
    2) Discretization whereby the continuous pixel parametrization is
       transformed into a discrete grating (Note that L2D is implemented here).
    3) Further optimization of the discrete grating by moving the grating
       edges.

    Args:
        obj: The objective function to minimize.
        monitors: List of monitors to keep track of.
        cont_iters: Number of iterations to run in continuous optimization.
        disc_iters: Number of iterations to run in discrete optimization.
        sim_space: Simulation space to use.
        min_feature: Minimum feature size in nanometers.
        cont_to_disc_factor: Discretize the continuous grating with feature size
            constraint of `min_feature * cont_to_disc_factor`.
            `cont_to_disc_factor > 1` gives discrete optimization more wiggle room.

    Returns:
        A list of transformations.
    """
    # First do continuous relaxation optimization.
    cont_param = optplan.PixelParametrization(
        simulation_space=sim_space,
        init_method=optplan.UniformInitializer(min_val=0, max_val=1),
    )
    trans_list = [
        optplan.Transformation(
            name="opt_cont",
            parametrization=cont_param,
            transformation=optplan.ScipyOptimizerTransformation(
                optimizer="L-BFGS-B",
                objective=obj,
                monitor_lists=optplan.ScipyOptimizerMonitorList(
                    callback_monitors=monitors,
                    start_monitors=monitors,
                    end_monitors=monitors,
                ),
                optimization_options=optplan.ScipyOptimizerOptions(maxiter=cont_iters),
            ),
        )
    ]
    # If true, do another round of continuous optimization with a discreteness bias.
    if DISCRETENESS_PENALTY:
        # Define parameters necessary to normaize discrete penalty term
        obj_val_param = optplan.Parameter(name="param_obj_final_val", initial_value=1.0)
        obj_val_param_abs = optplan.abs(obj_val_param)

        discrete_penalty_val = optplan.Parameter(
            name="param_discrete_penalty_val", initial_value=1.0
        )
        discrete_penalty_val_abs = optplan.abs(discrete_penalty_val)

        # Initial value of scaling is arbitrary and set for specific problem
        disc_scaling = optplan.Parameter(name="discrete_scaling", initial_value=5)

        normalization = disc_scaling * obj_val_param_abs / discrete_penalty_val_abs

        obj_disc = obj + optplan.DiscretePenalty() * normalization

        trans_list.append(
            optplan.Transformation(
                name="opt_cont_disc",
                parameter_list=[
                    optplan.SetParam(
                        parameter=obj_val_param,
                        function=obj,
                        parametrization=cont_param,
                    ),
                    optplan.SetParam(
                        parameter=discrete_penalty_val,
                        function=optplan.DiscretePenalty(),
                        parametrization=cont_param,
                    ),
                ],
                parametrization=cont_param,
                transformation=optplan.ScipyOptimizerTransformation(
                    optimizer="L-BFGS-B",
                    objective=obj_disc,
                    monitor_lists=optplan.ScipyOptimizerMonitorList(
                        callback_monitors=monitors,
                        start_monitors=monitors,
                        end_monitors=monitors,
                    ),
                    optimization_options=optplan.ScipyOptimizerOptions(
                        maxiter=cont_iters
                    ),
                ),
            )
        )

    # Discretize. Note we add a little bit of wiggle room by discretizing with
    # a slightly larger feature size that what our target is (by factor of
    # `cont_to_disc_factor`). This is to give the optimization a bit more wiggle
    # room later on.
    disc_param = optplan.GratingParametrization(
        simulation_space=sim_space, inverted=True
    )
    trans_list.extend(
        (
            optplan.Transformation(
                name="cont_to_disc",
                parametrization=disc_param,
                transformation=optplan.GratingEdgeFitTransformation(
                    parametrization=cont_param,
                    min_feature=cont_to_disc_factor * min_feature,
                ),
            ),
            optplan.Transformation(
                name="opt_disc",
                parametrization=disc_param,
                transformation=optplan.ScipyOptimizerTransformation(
                    optimizer="SLSQP",
                    objective=obj,
                    constraints_ineq=[
                        optplan.GratingFeatureConstraint(
                            min_feature_size=min_feature,
                            simulation_space=sim_space,
                            boundary_constraint_scale=1.0,
                        )
                    ],
                    monitor_lists=optplan.ScipyOptimizerMonitorList(
                        callback_monitors=monitors,
                        start_monitors=monitors,
                        end_monitors=monitors,
                    ),
                    optimization_options=optplan.ScipyOptimizerOptions(
                        maxiter=disc_iters
                    ),
                ),
            ),
        )
    )
    return trans_list


def view_opt_progress(save_folder: str, key="mon_power_1550") -> List[float]:
    """Shows the result of the optimization.

    Args:
        save_folder: Location where the log files are saved.
        key: for monitor
    """

    step = 1
    fp = pathlib.Path(save_folder) / f"step{step}.pkl"
    mon = []

    while fp.exists():
        log_data = pickle.loads(fp.read_bytes())
        mon.append(log_data["monitor_data"][key])
        step += 1
        fp = save_folder / f"step{step}.pkl"
    return mon


def view_opt_yaml(save_folder: str) -> None:
    """Shows the result of the optimization.

    This runs the auto-plotter to plot all the relevant data.
    See `examples/wdm2` IPython notebook for more details on how to process
    the optimization logs.

    Args:
        save_folder: Location where the log files are saved.
    """
    log_df = log_tools.create_log_data_frame(log_tools.load_all_logs(save_folder))
    monitor_descriptions = log_tools.load_from_yml(
        os.path.join(os.path.dirname(__file__), "monitor_spec.yml")
    )
    log_tools.plot_monitor_data(log_df, monitor_descriptions)


def view_opt_quick(save_folder: str) -> None:
    """Prints the current result of the optimization.

    Unlike `view_opt`, which plots fields and optimization trajectories,
    `view_opt_quick` prints out scalar monitors in the latest log file. This
    is useful for having a quick look into the state of the optimization.

    Args:
        save_folder: Location where the log files are saved.
    """
    with open(workspace.get_latest_log_file(save_folder), "rb") as fp:
        log_data = pickle.load(fp)
        for key, data in log_data["monitor_data"].items():
            if np.isscalar(data):
                print(f"{key}: {data.squeeze()}")


def resume_opt(save_folder: str) -> None:
    """Resumes a stopped optimization.

    This restarts an optimization that was stopped prematurely. Note that
    resuming an optimization will not lead the exact same results as if the
    optimization were finished the first time around.

    Args:
        save_folder: Location where log files are saved. It is assumed that
            the optimization plan is also saved there.
    """
    # Load the optimization plan.
    with open(os.path.join(save_folder, "optplan.json")) as fp:
        plan = optplan.loads(fp.read())

    # Run the plan with the `resume` flag to restart.
    problem_graph.run_plan(plan, ".", save_folder=save_folder, resume=True)


def gen_gds(save_folder: str, grating_len: float, wg_width: float) -> None:
    """Generates a GDS file of the grating.

    Args:
        save_folder: Location where log files are saved. It is assumed that
            the optimization plan is also saved there.
        grating_len: Length of the grating.
        wg_width: Width of the grating/bus waveguide.
    """
    # Load the optimization plan.
    with open(os.path.join(save_folder, "optplan.json")) as fp:
        plan = optplan.loads(fp.read())
    dx = plan.transformations[-1].parametrization.simulation_space.mesh.dx

    # Load the data from the latest log file.
    with open(workspace.get_latest_log_file(save_folder), "rb") as fp:
        log_data = pickle.load(fp)
        if log_data["transformation"] != plan.transformations[-1].name:
            raise ValueError("Optimization did not run until completion.")

        coords = log_data["parametrization"]["vector"] * dx

        if plan.transformations[-1].parametrization.inverted:
            coords = np.insert(coords, 0, 0, axis=0)
            coords = np.insert(coords, -1, grating_len, axis=0)

    # `coords` now contains the location of the grating edges. Now draw a
    # series of rectangles to represent the grating.
    grating_poly = [
        (
            (coords[i], -wg_width / 2),
            (coords[i], wg_width / 2),
            (coords[i + 1], wg_width / 2),
            (coords[i + 1], -wg_width / 2),
        )
        for i in range(0, len(coords), 2)
    ]
    return grating_poly


if __name__ == "__main__":
    # save_folder = pathlib.Path(__file__).parent / "demo"
    # mon = view_opt_progress(save_folder)
    # return
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("run", "view", "view_quick", "resume", "gen_gds"),
        help='Must be either "run" to run an optimization, "view" to '
        'view the results, "resume" to resume an optimization, or '
        '"gen_gds" to generate the grating GDS file.',
    )
    parser.add_argument("save_folder", help="Folder containing optimization logs.")

    grating_len = 12000
    wg_width = 12000

    args = parser.parse_args()
    if args.action == "run":
        run_opt(args.save_folder, grating_len=grating_len, wg_width=wg_width)
    elif args.action == "view":
        view_opt_progress(args.save_folder)
    elif args.action == "view_quick":
        view_opt_quick(args.save_folder)
    elif args.action == "resume":
        resume_opt(args.save_folder)
    elif args.action == "gen_gds":
        gen_gds(args.save_folder, grating_len=grating_len, wg_width=wg_width)
