"""Compute and write Sparameters using Meep in MPI."""

from __future__ import annotations

import multiprocessing
import pathlib
import pickle
import shlex
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional

import pydantic

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import logger, sparameters_path
from gdsfactory.pdk import get_layer_stack
from gdsfactory.simulation import port_symmetries
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_meep as get_sparameters_path,
)
from gdsfactory.simulation.gmeep.write_sparameters_meep import (
    remove_simulation_kwargs,
    settings_write_sparameters_meep,
)
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentSpec, PathType

core_materials = multiprocessing.cpu_count()

temp_dir_default = Path(sparameters_path) / "temp"


def _python() -> str:
    """Select correct python executable from current activated environment."""
    return sys.executable


@pydantic.validate_arguments
def write_sparameters_meep_mpi(
    component: ComponentSpec,
    layer_stack: Optional[LayerStack] = None,
    cores: int = core_materials,
    filepath: Optional[PathType] = None,
    dirpath: Optional[PathType] = None,
    temp_dir: Path = temp_dir_default,
    temp_file_str: str = "write_sparameters_meep_mpi",
    live_output: bool = False,
    overwrite: bool = False,
    wait_to_finish: bool = True,
    **kwargs,
) -> Path:
    """Write Sparameters using multiple cores and MPI and returns Sparameters filepath.

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
        dirpath: directory to store sparameters in CSV.
            Defaults to active Pdk.sparameters_path.
        layer_stack: contains layer to thickness, zmin and material.
            Defaults to active pdk.layer_stack.
        temp_dir: temporary directory to hold simulation files.
        temp_file_str: names of temporary files in temp_dir.
        live_output: stream output of mpirun command to file and print to console
            (meep verbosity still needs to be set separately).
        overwrite: overwrites stored simulation results.
        wait_to_finish: if True makes the function call blocking.

    Keyword Args:
        resolution: in pixels/um (30: for coarse, 100: for fine).
        port_symmetries: Dict to specify port symmetries, to save number of simulations.
        dirpath: directory to store Sparameters.
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
        port_margin: margin on each side of the port.
        distance_source_to_monitors: in (um) source goes before.
        port_source_offset: offset between source GDS port and source MEEP port.
        port_monitor_offset: offset between monitor GDS port and monitor MEEP port.

    Returns:
        filepath for sparameters CSV (wavelengths, s11a, o1@0,o2@0, ...)
            where `a` is the angle in radians and `m` the module.

    TODO:
        write stdout to file, maybe simulation logs too.

    """
    for setting in kwargs:
        if setting not in settings_write_sparameters_meep:
            raise ValueError(f"{setting!r} not in {settings_write_sparameters_meep}")

    component = gf.get_component(component)
    assert isinstance(component, Component)

    layer_stack = layer_stack or get_layer_stack()

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
    filepath_json = tempfile.with_suffix(".json")
    logger.info(f"Write {filepath_json!r}")

    layer_stack_json = layer_stack.json()
    filepath_json.write_text(layer_stack_json)

    parameters_file = tempfile.with_suffix(".pkl")
    with open(parameters_file, "wb") as outp:
        pickle.dump(settings, outp, pickle.HIGHEST_PROTOCOL)

    # Save component to disk through gds for gdstk compatibility
    component_file = tempfile.with_suffix(".gds")
    component.write_gds(component_file, with_metadata=True)

    # Write execution file
    script_lines = [
        "import pathlib\n",
        "import pickle\n",
        "from gdsfactory.simulation.gmeep import write_sparameters_meep\n\n",
        "from gdsfactory.read import import_gds\n",
        "from gdsfactory.technology import LayerStack\n\n",
        "if __name__ == '__main__':\n",
        f"\twith open(\"{parameters_file}\", 'rb') as inp:\n",
        "\t\tparameters_dict = pickle.load(inp)\n\n",
        f"\tcomponent = import_gds({str(component_file)!r}, read_metadata=True)\n",
        f"\tfilepath_json = pathlib.Path({str(filepath_json)!r})\n",
        "\tlayer_stack = LayerStack.parse_raw(filepath_json.read_text())\n",
        f"\twrite_sparameters_meep(component=component, overwrite={overwrite}, "
        f"layer_stack=layer_stack, filepath={str(filepath)!r},",
    ]
    script_lines.extend(f'\t\t{key} = parameters_dict["{key}"],\n' for key in settings)
    script_lines.append("\t)")

    script_file = tempfile.with_suffix(".py")
    with open(script_file, "w") as script_file_obj:
        script_file_obj.writelines(script_lines)
    command = f"mpirun -np {cores} {_python()} {script_file}"
    logger.info(command)
    logger.info(str(filepath))

    if live_output:
        import asyncio

        from gdsfactory.utils.async_utils import execute_and_stream_output

        asyncio.run(
            execute_and_stream_output(
                command, log_file_dir=temp_dir, log_file_str=temp_file_str
            )
        )
    else:
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


write_sparameters_meep_mpi_1x1 = partial(
    write_sparameters_meep_mpi, port_symmetries=port_symmetries.port_symmetries_1x1
)

write_sparameters_meep_mpi_1x1_bend90 = partial(
    write_sparameters_meep_mpi,
    ymargin_bot=3,
    ymargin=0,
    xmargin_right=3,
    port_symmetries=port_symmetries.port_symmetries_1x1,
)


if __name__ == "__main__":
    import numpy as np

    c1 = gf.components.straight(length=2.1)
    filepath = write_sparameters_meep_mpi(
        component=c1,
        # ymargin=3,
        cores=2,
        run=True,
        overwrite=True,
        live_output=True,
        # lazy_parallelism=True,
        lazy_parallelism=False,
        temp_dir="./test/",
        filepath="instance_dict.csv",
        resolution=20,
    )
    sp = np.load(filepath)
    print(list(sp.keys()))
