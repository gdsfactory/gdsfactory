"""SMF specs from photonics.byu.edu/FiberOpticConnectors.parts/images/smf28.pdf

MFD:

- 10.4 for Cband
- 9.2 for Oband

"""
import hashlib
import pathlib
import shlex
import shutil
import subprocess
import time
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import omegaconf
import pandas as pd

from gdsfactory.config import logger, sparameters_path
from gdsfactory.serialization import clean_value_json, clean_value_name
from gdsfactory.simulation.gmeep.get_simulation_grating_fiber import (
    get_simulation_grating_fiber,
)
from gdsfactory.types import PathType

nm = 1e-3
nSi = 3.48
nSiO2 = 1.44

Floats = Tuple[float, ...]


def fiber_ncore(fiber_numerical_aperture, fiber_nclad):
    return (fiber_numerical_aperture**2 + fiber_nclad**2) ** 0.5


def write_sparameters_grating(
    plot: bool = False,
    plot_contour: bool = False,
    animate: bool = False,
    overwrite: bool = False,
    dirpath: Optional[PathType] = sparameters_path,
    decay_by: float = 1e-3,
    verbosity: int = 0,
    ncores: int = 1,
    **settings,
) -> pd.DataFrame:
    """Write sparameter simulation results from grating coupler with fiber.

    Args:
        plot: plot simulation (do not run)
        plot_contour:
        animate:
        overwrite: overwrites simulation if found
        dirpath: directory path
        decay_by: field decay to stop simulation
        verbosity:
        ncores:

    Keyword Args:
        period: fiber grating period
        fill_factor:
        n_periods: number of periods
        widths: Optional list of widths. Overrides period, fill_factor, n_periods
        gaps: Optional list of gaps. Overrides period, fill_factor, n_periods
        etch_depth: grating etch depth
        fiber_angle_deg: fiber angle in degrees
        fiber_xposition: xposition
        fiber_core_diameter: fiber diameter
        fiber_numerical_aperture: NA
        fiber_nclad: fiber cladding index.
        ncore: fiber index core.
        nclad: cladding index top.
        nbox: box index.
        nsubstrate: index substrate.
        pml_thickness: pml_thickness (um)
        substrate_thickness: substrate_thickness (um)
        box_thickness: thickness for bottom cladding (um)
        core_thickness: core_thickness (um)
        top_clad_thickness: float = 2.0,
        air_gap_thickness: float = 1.0,
        fiber_thickness: float = 2.0,
        resolution: resolution pixels/um
        wavelength_min: min wavelength (um)
        wavelength_max: max wavelength (um)
        wavelength_points: wavelength points.
        eps_averaging: epsilon averaging.
        fiber_port_y_offset_from_air: y_offset from fiber to air (um).
        waveguide_port_x_offset_from_grating_start:
        fiber_port_x_size:
    """
    mp.verbosity(verbosity)

    settings = clean_value_json(settings)
    settings_string = clean_value_name(settings)
    settings_hash = hashlib.md5(settings_string.encode()).hexdigest()[:8]

    filename = f"fiber_{settings_hash}.yml"
    dirpath = dirpath or pathlib.Path(__file__).parent / "data"
    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(exist_ok=True, parents=True)
    filepath = dirpath / filename
    filepath_csv = filepath.with_suffix(".csv")
    filepath_mp4 = filepath.with_suffix(".mp4")

    if filepath_csv.exists() and not overwrite and not plot:
        logger.info(f"sparameters loaded from {str(filepath_csv)!r}")
        return pd.read_csv(filepath_csv)

    sim_dict = get_simulation_grating_fiber(**settings)
    sim = sim_dict["sim"]
    freqs = sim_dict["freqs"]
    start = time.time()

    if plot or plot_contour:
        eps_parameters = dict(contour=True) if plot_contour else None
        sim.plot2D(eps_parameters=eps_parameters)
        plt.show()
        return

    termination = [
        mp.stop_when_fields_decayed(
            dt=50,
            c=mp.Ez,
            pt=monitor.regions[0].center,
            decay_by=decay_by,
        )
        for monitor in [sim_dict["waveguide_monitor"], sim_dict["fiber_monitor"]]
    ]

    if animate:
        # Run while saving fields
        # sim.use_output_directory()
        animate = mp.Animate2D(
            sim,
            fields=mp.Ez,
            realtime=False,
            normalize=True,
            eps_parameters={"contour": True},
            field_parameters={
                "alpha": 0.8,
                "cmap": "RdBu",
                "interpolation": "none",
            },
            boundary_parameters={
                "hatch": "o",
                "linewidth": 1.5,
                "facecolor": "y",
                "edgecolor": "b",
                "alpha": 0.3,
            },
        )

        sim.run(mp.at_every(1, animate), until_after_sources=termination)
        animate.to_mp4(15, filepath_mp4)

    else:
        sim.run(until_after_sources=termination)

    # Extract mode information
    waveguide_monitor = sim_dict["waveguide_monitor"]
    waveguide_port_direction = sim_dict["waveguide_port_direction"]
    fiber_monitor = sim_dict["fiber_monitor"]
    fiber_angle_deg = sim_dict["fiber_angle_deg"]
    fcen = sim_dict["fcen"]
    wavelengths = 1 / freqs

    waveguide_mode = sim.get_eigenmode_coefficients(
        waveguide_monitor,
        [1],
        eig_parity=mp.ODD_Z,
        direction=waveguide_port_direction,
    )
    fiber_mode = sim.get_eigenmode_coefficients(
        fiber_monitor,
        [1],
        direction=mp.NO_DIRECTION,
        eig_parity=mp.ODD_Z,
        kpoint_func=lambda f, n: mp.Vector3(0, fcen * 1.45, 0).rotate(
            mp.Vector3(z=1), -1 * np.radians(fiber_angle_deg)
        ),  # Hardcoded index for now, pull from simulation eventually
    )
    end = time.time()

    a1 = waveguide_mode.alpha[:, :, 0].flatten()  # forward wave
    b1 = waveguide_mode.alpha[:, :, 1].flatten()  # backward wave

    # Since waveguide port is oblique, figure out forward and backward direction
    kdom_fiber = fiber_mode.kdom[0]
    idx = 1 - (kdom_fiber.y > 0) * 1

    a2 = fiber_mode.alpha[:, :, idx].flatten()  # forward wave
    # b2 = fiber_mode.alpha[:, :, 1 - idx].flatten()  # backward wave

    s11 = np.squeeze(b1 / a1)
    s12 = np.squeeze(a2 / a1)
    s22 = s11.copy()
    s21 = s12.copy()

    simulation = dict(
        settings=settings,
        compute_time_seconds=end - start,
        compute_time_minutes=(end - start) / 60,
    )
    filepath.write_text(omegaconf.OmegaConf.to_yaml(simulation))

    r = dict(s11=s11, s12=s12, s21=s21, s22=s22, wavelengths=wavelengths)
    keys = [key for key in r if key.startswith("s")]
    s = {f"{key}a": list(np.unwrap(np.angle(r[key].flatten()))) for key in keys} | {
        f"{key}m": list(np.abs(r[key].flatten())) for key in keys
    }

    s["wavelengths"] = wavelengths

    df = pd.DataFrame(s, index=wavelengths)
    df.to_csv(filepath_csv, index=False)
    return df


def write_sparameters_grating_mpi(
    instance: Dict,
    cores: int = 2,
    temp_dir: Optional[str] = None,
    temp_file_str: str = "write_sparameters_meep_mpi",
    verbosity: bool = False,
):
    """
    Given a Dict of write_sparameters_meep keyword arguments (the "instance"),
    launches a parallel simulation on `cores` cores
    Returns the subprocess Popen object

    Args
        instances (Dict): The keys must be parameters names of write_sparameters_meep, and entries the values
        cores (int): number of processors
        temp_dir (FilePath): temporary directory to hold simulation files
        temp_file_str (str): names of temporary files in temp_dir
        verbosity (bool): progress messages
    """

    # Save the component object to simulation for later retrieval
    temp_dir = temp_dir or pathlib.Path(__file__).parent / "temp"
    temp_dir = pathlib.Path(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    filepath = temp_dir / temp_file_str

    # Add parallelism info
    instance["ncores"] = cores

    # Write execution file
    script_lines = [
        "from gdsfactory.simulation.gmeep.write_sparameters_grating import write_sparameters_grating\n\n",
        'if __name__ == "__main__":\n\n',
        "\twrite_sparameters_grating(\n",
    ]
    for key in instance.keys():
        if isinstance(instance[key], str):
            parameter = f'"{instance[key]}"'
        else:
            parameter = instance[key]
        script_lines.append(f"\t\t{key} = {parameter},\n")
    script_lines.append("\t)")
    script_file = filepath.with_suffix(".py")
    with open(script_file, "w") as script_file_obj:
        script_file_obj.writelines(script_lines)
    # Exec string
    command = f"mpirun -np {cores} python {script_file}"

    # Launch simulation
    if verbosity:
        print(f"Launching: {command}")
    return subprocess.Popen(
        shlex.split(command),
        shell=False,
        stdin=None,
        stdout=None,
        stderr=None,
    )


def write_sparameters_grating_batch(
    instances,
    cores_per_instance: int = 2,
    total_cores: int = 4,
    temp_dir: Optional[str] = None,
    delete_temp_files: bool = False,
    verbosity: bool = False,
) -> None:
    """Given a tuple of write_sparameters_meep keyword arguments (instances)
    launches parallel simulations each simulation is assigned "cores_per_instance" cores
    A total of "total_cores" is assumed, if cores_per_instance * len(instances) > total_cores
    then the overflow will be performed serially

    Args
        instances: list of Dicts. The keys must be parameters names of write_sparameters_meep, and entries the values
        cores_per_instance (int): number of processors to assign to each instance
        total_cores (int): total number of cores to use
        temp_dir (FilePath): temporary directory to hold simulation files
        delete_temp_file (Boolean): whether to delete temp_dir when done
        verbosity: progress messages
    """
    # Save the component object to simulation for later retrieval
    temp_dir = temp_dir or pathlib.Path(__file__).parent / "temp"
    temp_dir = pathlib.Path(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)

    # Setup pools
    num_pools = int(np.ceil(cores_per_instance * len(instances) / total_cores))
    instances_per_pool = int(np.floor(total_cores / cores_per_instance))
    num_tasks = len(instances)

    if verbosity:
        print(f"Running parallel simulations over {num_tasks} instances")
        print(
            f"Using a total of {total_cores} cores with {cores_per_instance} cores per instance"
        )
        print(
            f"Tasks split amongst {num_pools} pools with up to {instances_per_pool} instances each."
        )

    i = 0
    # For each pool
    for j in range(num_pools):
        processes = []
        # For instance in the pool
        for k in range(instances_per_pool):
            # Flag to catch nonfull pools
            if i >= num_tasks:
                continue
            if verbosity:
                print(f"Task {k} of pool {j} is instance {i}")
            # Obtain current instance
            instance = instances[i]

            process = write_sparameters_grating_batch(
                instances=instance,
                cores_per_instance=cores_per_instance,
                temp_dir=temp_dir,
                verbosity=verbosity,
            )
            processes.append(process)

            # Increment task number
            i += 1

        # Wait for pool to end
        for process in processes:
            process.wait()

    if delete_temp_files:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # fiber_numerical_aperture = float(np.sqrt(1.44427 ** 2 - 1.43482 ** 2))
    # instance1 = dict(
    #     period=0.66,
    #     fill_factor=0.5,
    #     n_periods=50,
    #     etch_depth=70 * nm,
    #     fiber_angle_deg=10.0,
    #     fiber_xposition=0.0,
    #     fiber_core_diameter=9,
    #     fiber_numerical_aperture=fiber_numerical_aperture,
    #     fiber_nclad=1.43482,
    #     ncore=3.47,
    #     nclad=1.44,
    #     nbox=1.44,
    #     nsubstrate=3.47,
    #     pml_thickness=1.0,
    #     substrate_thickness=1.0,
    #     box_thickness=2.0,
    #     core_thickness=220 * nm,
    #     top_clad_thickness=2.0,
    #     air_gap_thickness=1.0,
    #     fiber_thickness=2.0,
    #     res=20,  # pixels/um
    #     wavelength_min=1.4,
    #     wavelength_max=1.7,
    #     wavelength_points=150,
    #     fiber_port_y_offset_from_air=1,
    #     waveguide_port_x_offset_from_grating_start=10,
    #     overwrite=True,
    #     verbosity=0,
    #     decay_by=1e-3,
    # )
    # instance2 = instance1.copy()
    # instance2["period"] = 0.5
    # write_sparameters_meep_batch(
    #     instances=[instance1, instance2],
    #     cores_per_instance=4,
    #     total_cores=8,
    #     verbosity=True,
    #     delete_temp_files=True,
    # )

    from gdsfactory.simulation.plot import plot_sparameters

    df = write_sparameters_grating(fiber_angle_deg=15)
    plot_sparameters(df)
