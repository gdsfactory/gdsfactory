from __future__ import annotations

import time

import numpy as np
import tidy3d as td
from omegaconf import OmegaConf

import gdsfactory as gf
from gdsfactory.config import logger
from gdsfactory.serialization import clean_value_json
from gdsfactory.simulation import port_symmetries
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_tidy3d as get_sparameters_path,
)
from gdsfactory.simulation.gtidy3d.get_results import _executor, get_results_batch
from gdsfactory.simulation.gtidy3d.get_simulation import get_simulation, plot_simulation
from gdsfactory.typings import (
    Any,
    ComponentSpec,
    Dict,
    List,
    Optional,
    PathType,
    Port,
    PortSymmetries,
    Tuple,
)


def parse_port_eigenmode_coeff(
    port_name: str, ports: Dict[str, Port], sim_data: td.SimulationData
) -> Tuple[np.ndarray]:
    """Given a port and eigenmode coefficient result, returns the coefficients \
    relative to whether the wavevector is entering or exiting simulation.

    Args:
        port_name: port name.
        ports: component_ref.ports.
        sim_data: simulation data.
    """
    # Direction of port (pointing away from the simulation)
    # Figure out if that is exiting the simulation or not
    # depending on the port orientation (assuming it's near PMLs)

    orientation = ports[port_name].orientation

    if orientation in [0, 90]:  # east
        direction_inp = "-"
        direction_out = "+"
    elif orientation in [180, 270]:  # west
        direction_inp = "+"
        direction_out = "-"
    else:
        raise ValueError(
            "Port orientation = {orientation} is not 0, 90, 180, or 270 degrees"
        )

    coeff_inp = sim_data.monitor_data[port_name].amps.sel(direction=direction_inp)
    coeff_out = sim_data.monitor_data[port_name].amps.sel(direction=direction_out)
    return coeff_inp.values.flatten(), coeff_out.values.flatten()


def get_wavelengths(port_name: str, sim_data: td.SimulationData) -> np.ndarray:
    coeff_inp = sim_data.monitor_data[port_name].amps.sel(direction="+")
    freqs = coeff_inp.f
    return td.constants.C_0 / freqs.values


def write_sparameters(
    component: ComponentSpec,
    port_symmetries: Optional[PortSymmetries] = None,
    port_source_names: Optional[List[str]] = None,
    dirpath: Optional[PathType] = None,
    run: bool = True,
    overwrite: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Get full sparameter matrix from a gdsfactory Component.

    Simulates each time using a different input port (by default, all of them)
    unless you specify port_symmetries.

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
        port_source_names: list of ports to excite. Defaults to all.
        port_symmetries: Dict to specify port symmetries, to save number of simulations
        dirpath: directory to store sparameters in npz.
            Defaults to active Pdk.sparameters_path.
        run: runs simulation, if False, only plots simulation.
        overwrite: overwrites stored Sparameter npz results.

    Keyword Args:
        port_extension: extend ports beyond the PML.
        layer_stack: contains layer to thickness, zmin and material.
            Defaults to active pdk.layer_stack.
        thickness_pml: PML thickness (um).
        xmargin: left/right distance from component to PML.
        xmargin_left: left distance from component to PML.
        xmargin_right: right distance from component to PML.
        ymargin: left/right distance from component to PML.
        ymargin_top: top distance from component to PML.
        ymargin_bot: bottom distance from component to PML.
        zmargin: thickness for cladding above and below core.
        clad_material: material for cladding.
        port_margin: margin on each side of the port.
        distance_source_to_monitors: in (um) source goes before monitors.
        wavelength_start: in (um).
        wavelength_stop: in (um).
        wavelength_points: in (um).
        plot_modes: plot source modes.
        num_modes: number of modes to plot.
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
        grid_spec: defaults to automatic td.GridSpec.auto(wavelength=wavelength)
            td.GridSpec.uniform(dl=20*nm)
            td.GridSpec(
                grid_x = td.UniformGrid(dl=0.04),
                grid_y = td.AutoGrid(min_steps_per_wvl=20),
                grid_z = td.AutoGrid(min_steps_per_wvl=20),
                wavelength=wavelength,
                override_structures=[refine_box]
            )
        dilation: float = 0.0
            Dilation of the polygon in the base by shifting each edge along its
            normal outwards direction by a distance;
            a negative value corresponds to erosion.
        sidewall_angle_deg : float = 0
            Angle of the sidewall.
            ``sidewall_angle=0`` (default) specifies vertical wall,
            while ``0<sidewall_angle_deg<90`` for the base to be larger than the top.

    """
    component = gf.get_component(component)
    filepath = get_sparameters_path(
        component=component,
        dirpath=dirpath,
        **kwargs,
    )
    filepath_sim_settings = filepath.with_suffix(".yml")
    if filepath.exists() and not overwrite and run:
        logger.info(f"Simulation loaded from {filepath!r}")
        return dict(np.load(filepath))

    port_symmetries = port_symmetries or {}
    component_ref = component.ref()
    ports = component_ref.ports
    port_names = [port.name for port in list(ports.values())]

    sims = []
    sp = {}

    port_source_names = port_source_names or port_names

    for port_name in port_source_names:
        if port_name not in port_symmetries:
            sim = get_simulation(component, port_source_name=port_name, **kwargs)
            sims.append(sim)

    if not run:
        sim = sims[0]
        plot_simulation(sim)
        return sp

    start = time.time()
    batch_data = get_results_batch(sims)

    def get_sparameter(
        port_name_source: str,
        sim_data: td.SimulationData,
        port_symmetries=port_symmetries,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Return Component sparameter for a particular port Index n.

        Args:
            port_name: source port name.
            sim_data: simulation data.
            port_symmetries: to save simulations.
            kwargs: simulation settings.
        """
        source_entering, source_exiting = parse_port_eigenmode_coeff(
            port_name=port_name_source, ports=component_ref.ports, sim_data=sim_data
        )

        for port_name in port_names:
            monitor_entering, monitor_exiting = parse_port_eigenmode_coeff(
                port_name=port_name, ports=ports, sim_data=sim_data
            )
            sij = monitor_exiting / source_entering
            key = f"{port_name}@0,{port_name_source}@0"
            sp[key] = sij
            sp["wavelengths"] = get_wavelengths(port_name=port_name, sim_data=sim_data)

        if bool(port_symmetries):
            for key, symmetries in port_symmetries.items():
                for sym in symmetries:
                    if key in sp:
                        sp[sym] = sp[key]

        return sp

    for port_source_name, (_sim_name, sim_data) in zip(
        port_source_names, batch_data.items()
    ):
        sp.update(get_sparameter(port_source_name, sim_data))

    end = time.time()
    np.savez_compressed(filepath, **sp)
    kwargs.update(compute_time_seconds=end - start)
    kwargs.update(compute_time_minutes=(end - start) / 60)
    filepath_sim_settings.write_text(OmegaConf.to_yaml(clean_value_json(kwargs)))
    logger.info(f"Write simulation results to {str(filepath)!r}")
    logger.info(f"Write simulation settings to {str(filepath_sim_settings)!r}")
    return sp


def write_sparameters_batch(jobs: List[Dict[str, Any]], **kwargs) -> List[np.ndarray]:
    """Returns Sparameters for a list of write_sparameters_grating_coupler kwargs \
            where it runs each simulation in parallel.

    Args:
        jobs: list of kwargs for write_sparameters_grating_coupler.
        kwargs: simulation settings.
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
    c = gf.c.straight()
    c = gf.components.mmi1x2()
    sp = write_sparameters(c, is_3d=True, port_source_names=None, overwrite=False)
    sim.plot.plot_sparameters(sp)

    # t = sp.o1@0,o2@0
    # print(f"Transmission = {t}")
    # cs = [gf.c.straight(length=1.11 + i) for i in [1, 2]]
    # sps = write_sparameters_batch_1x1(cs)
