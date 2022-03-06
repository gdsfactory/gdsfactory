import re
import time

import numpy as np
import pandas as pd
import tidy3d as td
from omegaconf import OmegaConf
from tqdm import tqdm

import gdsfactory as gf
from gdsfactory.config import logger, sparameters_path
from gdsfactory.serialization import clean_value_json
from gdsfactory.simulation import port_symmetries
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_tidy3d as get_sparameters_path,
)
from gdsfactory.simulation.gtidy3d.get_results import _executor, get_results
from gdsfactory.simulation.gtidy3d.get_simulation import get_simulation
from gdsfactory.types import (
    Any,
    Component,
    ComponentOrFactory,
    Dict,
    List,
    Optional,
    PathType,
    PortSymmetries,
)


def parse_port_eigenmode_coeff(port_index: int, ports, sim_data: td.SimulationData):
    """Given a port and eigenmode coefficient result, returns the coefficients
    relative to whether the wavevector is entering or exiting simulation

    Args:
        port_index: index of port
        ports: component_ref.ports
        sim_data: simulation data
    """
    if f"o{port_index}" not in ports:
        raise ValueError(
            f"port = 'o{port_index}' not in {list(ports.keys())}. "
            "You can rename ports with Component.auto_rename_ports()"
        )

    # Direction of port (pointing away from the simulation)
    # Figure out if that is exiting the simulation or not
    # depending on the port orientation (assuming it's near PMLs)

    orientation = ports[f"o{port_index}"].orientation
    if orientation == 0:  # east
        direction_inp = "-"
        direction_out = "+"
    elif orientation == 90:  # north
        direction_inp = "-"
        direction_out = "+"
    elif orientation == 180:  # west
        direction_inp = "+"
        direction_out = "-"
    elif orientation == 270:  # south
        direction_inp = "+"
        direction_out = "-"
    else:
        ValueError("Port orientation = {orientation} is not 0, 90, 180, or 270 degrees")

    coeff_inp = sim_data.monitor_data[f"o{port_index}"].amps.sel(
        direction=direction_inp
    )
    coeff_out = sim_data.monitor_data[f"o{port_index}"].amps.sel(
        direction=direction_out
    )
    return coeff_inp.values.flatten(), coeff_out.values.flatten()


def get_wavelengths(port_index, sim_data: td.SimulationData):
    coeff_inp = sim_data.monitor_data[f"o{port_index}"].amps.sel(direction="+")
    freqs = coeff_inp.f
    return td.constants.C_0 / freqs.values


def write_sparameters(
    component: ComponentOrFactory,
    port_symmetries: Optional[PortSymmetries] = None,
    dirpath: PathType = sparameters_path,
    overwrite: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Get full sparameter matrix from a gdsfactory Component.
    Simulates each time using a different input port (by default, all of them)
    unless you specify port_symmetries:

    port_symmetries = {"o1":
            {
                "s11": ["s22","s33","s44"],
                "s21": ["s21","s34","s43"],
                "s31": ["s13","s24","s42"],
                "s41": ["s14","s23","s32"],
            }
        }
    - Only simulations using the outer key port names will be run
    - The associated value is another dict whose keys are the S-parameters computed
        when this source is active
    - The values of this inner Dict are lists of s-parameters whose values are copied

    Args:
        component: to simulate.
        port_symmetries: Dict to specify port symmetries, to save number of simulations
        dirpath: directory to store sparameters in CSV.
        overwrite: overwrites stored Sparameter CSV results.

    Keyword Args:
        port_extension: extend ports beyond the PML.
        layer_stack: contains layer numbers (int, int) to thickness, zmin
        thickness_pml: PML thickness (um).
        xmargin: left/right distance from component to PML.
        xmargin_left: left distance from component to PML.
        xmargin_right: right distance from component to PML.
        ymargin: left/right distance from component to PML.
        ymargin_top: top distance from component to PML.
        ymargin_bot: bottom distance from component to PML.
        zmargin: thickness for cladding above and below core.
        clad_material: material for cladding.
        port_source_name: input port name.
        port_margin: margin on each side of the port.
        distance_source_to_monitors: in (um) source goes before monitors.
        resolution: in pixels/um (20: for coarse, 120: for fine)
        wavelength_start: in (um).
        wavelength_stop: in (um).
        wavelength_points: in (um).
        plot_modes: plot source modes.
        num_modes: number of modes to plot
        run_time_ps: make sure it's sufficient for the fields to decay.
            defaults to 10ps and counts on automatic shutoff to stop earlier if needed.
        dispersive: False uses constant refractive index materials.
            True adds wavelength depending materials.
            Dispersive materials require more computation.
        material_name_to_tidy3d_index: not dispersive materials have a constant index.
        material_name_to_tidy3d_name: dispersive materials have a wavelength
            dependent index. Maps layer_stack names with tidy3d material database names.
        is_3d: if False, does not consider Z dimension for faster simulations.
        with_all_monitors: True adds field monitor which increases results file size.

    """
    component = component() if callable(component) else component
    filepath = get_sparameters_path(
        component=component,
        dirpath=dirpath,
        **kwargs,
    )
    filepath_sim_settings = filepath.with_suffix(".yml")
    if filepath.exists() and not overwrite:
        logger.info(f"Simulation loaded from {filepath!r}")
        return pd.read_csv(filepath)

    port_symmetries = port_symmetries or {}
    monitor_indices = []
    source_indices = []
    component_ref = component.ref()

    for port_name in component_ref.ports.keys():
        if component_ref.ports[port_name].port_type == "optical":
            monitor_indices.append(re.findall("[0-9]+", port_name)[0])
    if bool(port_symmetries):  # user-specified
        for port_name in port_symmetries.keys():
            source_indices.append(re.findall("[0-9]+", port_name)[0])
    else:  # otherwise cycle through all
        source_indices = monitor_indices

    num_sims = len(port_symmetries.keys()) or len(source_indices)
    sp = {}

    def get_sparameter(
        n: int,
        component: Component,
        port_symmetries=port_symmetries,
        monitor_indices=monitor_indices,
        **kwargs,
    ) -> np.ndarray:
        """Return Component sparameter for a particular port Index n

        Args:
            n: port_index
            component:
            port_symmetries:
            monitor_indices:
            kwargs: simulation settings

        """
        sim = get_simulation(
            component, port_source_name=f"o{monitor_indices[n]}", **kwargs
        )
        sim_data = get_results(sim)
        sim_data = sim_data.result()
        source_entering, source_exiting = parse_port_eigenmode_coeff(
            monitor_indices[n], component_ref.ports, sim_data
        )

        for monitor_index in monitor_indices:
            j = monitor_indices[n]
            i = monitor_index
            if monitor_index == monitor_indices[n]:
                sii = source_exiting / source_entering

                siia = np.unwrap(np.angle(sii))
                siim = np.abs(sii)

                sp[f"s{i}{i}a"] = siia
                sp[f"s{i}{i}m"] = siim
            else:
                monitor_entering, monitor_exiting = parse_port_eigenmode_coeff(
                    monitor_index, component_ref.ports, sim_data
                )
                sij = monitor_exiting / source_entering
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)
                sp[f"s{i}{j}a"] = sija
                sp[f"s{i}{j}m"] = sijm
                sij = monitor_entering / source_entering
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)

        if bool(port_symmetries) is True:
            for key in port_symmetries[f"o{monitor_indices[n]}"].keys():
                values = port_symmetries[f"o{monitor_indices[n]}"][key]
                for value in values:
                    sp[f"{value}m"] = sp[f"{key}m"]
                    sp[f"{value}a"] = sp[f"{key}a"]

        sp["wavelengths"] = get_wavelengths(port_index=monitor_index, sim_data=sim_data)
        return sp

    start = time.time()

    # Compute each Sparameter on a separate thread
    sparameters = [
        _executor.submit(
            get_sparameter, n, component, port_symmetries, monitor_indices, **kwargs
        )
        for n in range(num_sims)
    ]

    for sparameter in tqdm(sparameters):
        sp.update(sparameter.result())

    end = time.time()
    df = pd.DataFrame(sp)
    df.to_csv(filepath, index=False)
    kwargs.update(compute_time_seconds=end - start)
    kwargs.update(compute_time_minutes=(end - start) / 60)

    filepath_sim_settings.write_text(OmegaConf.to_yaml(clean_value_json(kwargs)))
    logger.info(f"Write simulation results to {str(filepath)!r}")
    logger.info(f"Write simulation settings to {str(filepath_sim_settings)!r}")
    return df


def write_sparameters_batch(jobs: List[Dict[str, Any]], **kwargs) -> List[pd.DataFrame]:
    """Returns Sparameters for a list of write_sparameters_grating_coupler
    kwargs where it runs each simulation in paralell.

    Args:
        jobs: list of kwargs for write_sparameters_grating_coupler
        kwargs: simulation settings

    """
    sp = [_executor.submit(write_sparameters, **job, **kwargs) for job in jobs]
    return [spi.result() for spi in sp]


write_sparameters_1x1 = gf.partial(
    write_sparameters, port_symmetries=port_symmetries.port_symmetries_1x1
)
write_sparameters_crossing = gf.partial(
    write_sparameters, port_symmetries=port_symmetries.port_symmetries_crossing
)

write_sparameters_batch_1x1 = gf.partial(
    write_sparameters_batch, port_symmetries=port_symmetries.port_symmetries_1x1
)


if __name__ == "__main__":
    import gdsfactory as gf
    import gdsfactory.simulation as sim

    # c = gf.components.straight(length=2.1)
    c = gf.components.mmi2x2()
    df = write_sparameters(c, is_3d=False, overwrite=False)
    sim.plot.plot_sparameters(df)

    # t = df.s12m
    # print(f"Transmission = {t}")
    # cs = [gf.c.straight(length=1.11 + i) for i in [1, 2]]
    # dfs = write_sparameters_batch_1x1(cs)
