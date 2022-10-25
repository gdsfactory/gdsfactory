import time
from typing import Any, Dict, Optional

import numpy as np
import tidy3d as td
from omegaconf import OmegaConf

import gdsfactory as gf
from gdsfactory.config import logger
from gdsfactory.serialization import clean_value_json
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_tidy3d as get_sparameters_path,
)
from gdsfactory.simulation.gtidy3d.get_results import _executor, get_results
from gdsfactory.simulation.gtidy3d.get_simulation_grating_coupler import (
    get_simulation_grating_coupler,
)
from gdsfactory.types import Component, ComponentSpec, List, PathType


def write_sparameters_grating_coupler(
    component: ComponentSpec,
    dirpath: Optional[PathType] = None,
    overwrite: bool = False,
    port_waveguide_name: str = "o1",
    fiber_port_name: str = "vertical_te",
    **kwargs,
) -> np.ndarray:
    """Get sparameter matrix from a gdsfactory grating coupler.

    Assumes grating coupler waveguide port is facing to the left (west).

    TODO: add a fiber model (more realistic than a gaussian_beam).

    Args:
        component: grating coupler gdsfactory Component to simulate.
        dirpath: directory to store sparameters in npz.
            Defaults to active Pdk.sparameters_path.
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
        box_material: for bottom cladding.
        substrate_material: for substrate.
        box_thickness: bottom cladding thickness in (um).
        substrate_thickness: (um).
        port_waveguide_name: input port name.
        port_margin: margin on each side of the port.
        distance_source_to_monitors: in (um) source goes before monitors.
        port_waveguide_offset: mode solver workaround.
            positive moves source forward, negative moves source backward.
        wavelength: source center wavelength (um).
            if None takes mean between wavelength_start, wavelength_stop.
        wavelength_start: in (um).
        wavelength_stop: in (um).
        wavelength_points: number of wavelengths.
        plot_modes: plot source modes.
        num_modes: number of modes to plot.
        run_time_ps: make sure it's sufficient for the fields to decay.
            defaults to 10ps and counts on the automatic shutoff
            to stop earlier if needed.
        fiber_port_name: from component ports.
        fiber_xoffset: fiber center xoffset to fiber_port_name.
        fiber_z: fiber zoffset from grating zmax.
        fiber_mfd: fiber mode field diameter (um).
        fiber_angle_deg: fiber_angle in degrees with respect to normal.
        dispersive: False uses constant refractive index materials.
            True adds wavelength depending materials.
            Dispersive materials require more computation.
        material_name_to_tidy3d_index: not dispersive materials have a constant index.
        material_name_to_tidy3d_name: dispersive materials have a wavelength.
            dependent index. Maps layer_stack names with tidy3d material database names.
        is_3d: True by default runs in 3D.
        with_all_monitors: stores all monitor fields.

    """
    component = gf.get_component(component)
    assert isinstance(component, Component)

    filepath = get_sparameters_path(
        component=component,
        dirpath=dirpath,
        **kwargs,
    )
    filepath_sim_settings = filepath.with_suffix(".yml")
    if filepath.exists():
        if overwrite:
            filepath.unlink()

        else:
            logger.info(f"Simulation loaded from {filepath!r}")
            return np.load(filepath)
    start = time.time()
    sim = get_simulation_grating_coupler(
        component,
        fiber_port_name=fiber_port_name,
        port_waveguide_name=port_waveguide_name,
        **kwargs,
    )
    sim_data = get_results(sim)
    sim_data = sim_data.result()

    direction_inp = "+"
    monitor_entering = (
        sim_data.monitor_data["waveguide"]
        .amps.sel(direction=direction_inp)
        .values.flatten()
    )
    direction_out = "-"
    monitor_exiting = (
        sim_data.monitor_data["waveguide"]
        .amps.sel(direction=direction_out)
        .values.flatten()
    )
    r = monitor_entering / monitor_exiting
    t = monitor_exiting

    freqs = sim_data.monitor_data["waveguide"].amps.sel(direction="+").f
    port_name_input = port_waveguide_name
    port_name_output = fiber_port_name

    key = f"s{port_name_input}@0,{port_name_input}@0"
    sp = {"wavelengths": td.constants.C_0 / freqs.values, key: r}
    key = f"s{port_name_output}@0,{port_name_output}@0"
    sp[key] = r

    key = f"s{port_name_input}@0,{port_name_output}@0"
    sp[key] = t

    key = f"s{port_name_output}@0,{port_name_input}@0"
    sp[key] = t

    end = time.time()
    np.savez_compressed(filepath, sp)
    kwargs.update(compute_time_seconds=end - start)
    kwargs.update(compute_time_minutes=(end - start) / 60)

    filepath_sim_settings.write_text(OmegaConf.to_yaml(clean_value_json(kwargs)))
    logger.info(f"Write simulation results to {str(filepath)!r}")
    logger.info(f"Write simulation settings to {str(filepath_sim_settings)!r}")
    return sp


def write_sparameters_grating_coupler_batch(
    jobs: List[Dict[str, Any]], **kwargs
) -> List[np.ndarray]:
    """Returns Sparameters for a list of write_sparameters_grating_coupler settings.

    Each simulation runs in parallel.

    Args:
        jobs: list of kwargs for write_sparameters_grating_coupler.
        kwargs: simulation settings.

    """
    sp = [
        _executor.submit(write_sparameters_grating_coupler, **job, **kwargs)
        for job in jobs
    ]
    return [spi.result() for spi in sp]


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import gdsfactory.simulation as sim

    c = gf.components.grating_coupler_elliptical_lumerical()  # inverse design grating
    sp = write_sparameters_grating_coupler(
        c,
        is_3d=False,
        fiber_angle_deg=-5,
        fiber_xoffset=+2,
    )

    # sim.plot.plot_sparameters(sp)

    # c = gf.components.grating_coupler_elliptical_arbitrary(
    #     widths=[0.343] * 25,
    #     gaps=[0.345] * 25,
    # )
    # sp = write_sparameters_grating_coupler(c, is_3d=False)
    # t = sp.o1@0,o2@0
    # print(f"Transmission = {t}")

    # plt.plot(sp.wavelengths, sp.o1@0,o2@0)
    # plt.show()
