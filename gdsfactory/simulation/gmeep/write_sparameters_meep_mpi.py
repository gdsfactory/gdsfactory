"""Compute and write Sparameters using Meep in MPI."""

import multiprocessing
import pathlib
import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pydantic

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import logger, sparameters_path
from gdsfactory.simulation import port_symmetries
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_meep as get_sparameters_path,
)
from gdsfactory.simulation.gmeep.write_sparameters_meep import (
    remove_simulation_kwargs,
    settings_write_sparameters_meep,
)
from gdsfactory.tech import LAYER_STACK, LayerStack
from gdsfactory.types import ComponentSpec

ncores = multiprocessing.cpu_count()

temp_dir_default = Path(sparameters_path) / "temp"


@pydantic.validate_arguments
def write_sparameters_meep_mpi(
    component: ComponentSpec,
    layer_stack: LayerStack = LAYER_STACK,
    cores: int = ncores,
    filepath: Optional[Path] = None,
    dirpath: Path = sparameters_path,
    temp_dir: Path = temp_dir_default,
    temp_file_str: str = "write_sparameters_meep_mpi",
    overwrite: bool = False,
    wait_to_finish: bool = True,
    **kwargs,
) -> Path:
    """Write Sparameters using multiple cores and MPI
    and returns Sparameters CSV filepath.

    Simulates each time using a different input port (by default, all of them)
    unless you specify port_symmetries:

    checks stderror and kills MPI job if there is any stderror message

    port_symmetries = {"o1":
            {
                "s11": ["s22","s33","s44"],
                "s21": ["s21","s34","s43"],
                "s31": ["s13","s24","s42"],
                "s41": ["s14","s23","s32"],
            }
        }

    Args:
        component: gdsfactory Component.
        cores: number of processors.
        filepath: to store pandas Dataframe with Sparameters in CSV format.
            Defaults to dirpath/component_.csv.
        dirpath: directory to store Sparameters.
        layer_stack: with thickness and material information.
        temp_dir: temporary directory to hold simulation files.
        temp_file_str: names of temporary files in temp_dir.
        overwrite: overwrites stored simulation results.
        wait_to_finish: if True makes the function call blocking.

    Keyword Args:
        resolution: in pixels/um (30: for coarse, 100: for fine).
        port_symmetries: Dict to specify port symmetries, to save number of simulations.
        dirpath: directory to store Sparameters.
        layer_stack: LayerStack class.
        port_margin: margin on each side of the port.
        port_monitor_offset: offset between monitor GDS port and monitor MEEP port.
        port_source_offset: offset between source GDS port and source MEEP port.
        filepath: to store pandas Dataframe with Sparameters in CSV format.
        animate: saves a MP4 images of the simulation for inspection, and also
            outputs during computation. The name of the file is the source index.
        lazy_parallelism: toggles the flag "meep.divide_parallel_processes" to
            perform the simulations with different sources in parallel.
        dispersive: use dispersive models for materials (requires higher resolution).
        xmargin: left and right distance from component to PML.
        xmargin_left: west distance from component to PML.
        xmargin_right: east distance from component to PML.
        ymargin: top and bottom distance from component to PML.
        ymargin_top: north distance from component to PML.
        ymargin_bot: south distance from component to PML.
        extend_ports_length: to extend ports beyond the PML.
        layer_stack: Dict of layer number (int, int) to thickness (um).
        zmargin_top: thickness for cladding above core.
        zmargin_bot: thickness for cladding below core.
        tpml: PML thickness (um).
        clad_material: material for cladding.
        is_3d: if True runs in 3D.
        wavelength_start: wavelength min (um).
        wavelength_stop: wavelength max (um).
        wavelength_points: wavelength steps.
        dfcen: delta frequency.
        port_source_name: input port name.
        port_field_monitor_name: for monitor field decay.
        port_margin: margin on each side of the port.
        distance_source_to_monitors: in (um) source goes before.
        port_source_offset: offset between source GDS port and source MEEP port.
        port_monitor_offset: offset between monitor GDS port and monitor MEEP port.

    Returns:
        filepath for sparameters CSV (wavelengths, s11a, s12m, ...)
            where `a` is the angle in radians and `m` the module

    TODO:
        write stdout to file, maybe simulation logs too
    """
    for setting in kwargs.keys():
        if setting not in settings_write_sparameters_meep:
            raise ValueError(f"{setting} not in {settings_write_sparameters_meep}")

    component = gf.get_component(component)
    assert isinstance(component, Component)

    settings = remove_simulation_kwargs(kwargs)
    filepath = filepath or get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_stack=layer_stack,
        **settings,
    )
    filepath = pathlib.Path(filepath)
    if filepath.exists() and not overwrite:
        logger.info(f"Simulation {filepath!r} already exists")
        return filepath

    if filepath.exists() and overwrite:
        filepath.unlink()

    # Save all the simulation arguments for later retrieval
    temp_dir.mkdir(exist_ok=True, parents=True)
    tempfile = temp_dir / temp_file_str
    parameters_file = tempfile.with_suffix(".pkl")
    kwargs.update(filepath=str(filepath))

    parameters_dict = {
        "component": component,
        "layer_stack": layer_stack,
        "overwrite": overwrite,
    }

    # Loop over kwargs
    for key in kwargs.keys():
        parameters_dict[key] = kwargs[key]

    with open(parameters_file, "wb") as outp:
        pickle.dump(parameters_dict, outp, pickle.HIGHEST_PROTOCOL)

    # Write execution file
    script_lines = [
        "import pickle\n",
        "from gdsfactory.simulation.gmeep import write_sparameters_meep\n\n",
        'if __name__ == "__main__":\n\n',
        f"\twith open(\"{parameters_file}\", 'rb') as inp:\n",
        "\t\tparameters_dict = pickle.load(inp)\n\n" "\twrite_sparameters_meep(\n",
    ]
    script_lines.extend(
        f'\t\t{key} = parameters_dict["{key}"],\n' for key in parameters_dict
    )

    script_lines.append("\t)")
    script_file = tempfile.with_suffix(".py")
    with open(script_file, "w") as script_file_obj:
        script_file_obj.writelines(script_lines)
    command = f"mpirun -np {cores} python {script_file}"
    logger.info(command)
    logger.info(str(filepath))

    with subprocess.Popen(
        shlex.split(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        print(proc.stdout.read().decode())
        print(proc.stderr.read().decode())
        sys.stdout.flush()
        sys.stderr.flush()

    if wait_to_finish and not proc.stderr:
        while not filepath.exists():
            print(proc.stdout.read().decode())
            print(proc.stderr.read().decode())
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(1)

    return filepath


write_sparameters_meep_mpi_1x1 = gf.partial(
    write_sparameters_meep_mpi, port_symmetries=port_symmetries.port_symmetries_1x1
)

write_sparameters_meep_mpi_1x1_bend90 = gf.partial(
    write_sparameters_meep_mpi,
    ymargin_bot=3,
    ymargin=0,
    xmargin_right=3,
    port_symmetries=port_symmetries.port_symmetries_1x1,
)


if __name__ == "__main__":
    c1 = gf.components.straight(length=2.1)
    filepath = write_sparameters_meep_mpi(
        component=c1,
        # ymargin=3,
        cores=2,
        run=True,
        overwrite=True,
        # lazy_parallelism=True,
        lazy_parallelism=False,
        # filepath="instance_dict.csv",
    )
