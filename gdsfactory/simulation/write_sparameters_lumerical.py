"""Write Sparameters with Lumerical FDTD."""

import dataclasses
import time
from pathlib import Path
from typing import Optional

import numpy as np
import omegaconf
import pandas as pd

import gdsfactory as gf
from gdsfactory.config import __version__, logger
from gdsfactory.simulation.get_sparameters_path import get_sparameters_path
from gdsfactory.tech import (
    LAYER_STACK,
    SIMULATION_SETTINGS,
    LayerStack,
    SimulationSettings,
)
from gdsfactory.types import ComponentOrFactory

run_false_warning = """
You have passed run=False to debug the simulation

run=False returns the simulation session for you to debug and make sure it is correct

To compute the Sparameters you need to pass run=True
"""

MATERIAL_NAME_TO_LUMERICAL = {
    "si": "Si (Silicon) - Palik",
    "sio2": "SiO2 (Glass) - Palik",
    "sin": "Si3N4 (Silicon Nitride) - Phillip",
}


def write_sparameters_lumerical(
    component: ComponentOrFactory,
    session: Optional[object] = None,
    run: bool = True,
    overwrite: bool = False,
    dirpath: Path = gf.CONFIG["sparameters"],
    layer_stack: LayerStack = LAYER_STACK,
    simulation_settings: SimulationSettings = SIMULATION_SETTINGS,
    **settings,
) -> pd.DataFrame:
    r"""Returns and writes component Sparameters using Lumerical FDTD.

    If simulation exists it returns the Sparameters directly unless overwrite=True
    which forces a re-run of the simulation


    Writes Sparameters both in .CSV and .DAT (interconnect format) as well as
    simulation settings in .YAML

    In the CSV format you can see `S12m` where `m` stands for magnitude
    and `S12a` where `a` stands for angle in radians

    Your components need to have ports, that will extend over the PML.

    .. image:: https://i.imgur.com/dHAzZRw.png

    For your Fab technology you can overwrite

    - Simulation Settings
    - dirpath
    - layerStack

    converts gdsfactory units (um) to Lumerical units (m)

    Disclaimer: This function tries to extract Sparameters automatically
    is hard to make a function that will fit all your possible simulation settings.
    You can use this function as an inspiration to create your own.

    Args:
        component: Component to simulate
        session: you can pass a session=lumapi.FDTD() or it will create one
        run: True runs Lumerical, False only draws simulation
        overwrite: run even if simulation results already exists
        dirpath: where to store the Sparameters
        layer_stack: layer_stack
        simulation_settings: dataclass with all simulation_settings
        settings: overwrite any simulation settings
            background_material: for the background
            port_margin: on both sides of the port width (um)
            port_height: port height (um)
            port_extension: port extension (um)
            mesh_accuracy: 2 (1: coarse, 2: fine, 3: superfine)
            zmargin: for the FDTD region (um)
            ymargin: for the FDTD region (um)
            xmargin: for the FDTD region (um)
            wavelength_start: 1.2 (um)
            wavelength_stop: 1.6 (um)
            wavelength_points: 500
            simulation_time: (s) related to max path length 3e8/2.4*10e-12*1e6 = 1.25mm
            simulation_temperature: in kelvin (default = 300)
            frequency_dependendent_profile: computes mode profiles for different wavelengths
            field_profile_samples: number of wavelengths to compute field profile


    .. code::

         top view
              ________________________________
             |                               |
             | xmargin                       | port_extension
             |<------>          port_margin ||<-->
          ___|___________          _________||___
             |           \        /          |
             |            \      /           |
             |             ======            |
             |            /      \           |
          ___|___________/        \__________|___
             |   |                           |
             |   |ymargin                    |
             |   |                           |
             |___|___________________________|

        side view
              ________________________________
             |                               |
             |                               |
             |                               |
             |ymargin                        |
             |<---> _____         _____      |
             |     |     |       |     |     |
             |     |     |       |     |     |
             |     |_____|       |_____|     |
             |       |                       |
             |       |                       |
             |       |zmargin                |
             |       |                       |
             |_______|_______________________|



    Return:
        Sparameters pandas DataFrame (wavelengths, s11m, s11a, s12a ...)
        suffix `a` for angle in radians and `m` for module

    """
    component = component() if callable(component) else component
    sim_settings = dataclasses.asdict(simulation_settings)

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    layer_to_material = layer_stack.get_layer_to_material()

    if hasattr(component.info, "simulation_settings"):
        sim_settings.update(component.info.simulation_settings)
        logger.info(
            "Updating {component.name} sim settings {component.simulation_settings}"
        )
    for setting in settings.keys():
        if setting not in sim_settings:
            raise ValueError(
                f"Invalid setting `{setting}` not in ({list(sim_settings.keys())})"
            )

    sim_settings.update(**settings)
    ss = SimulationSettings(**sim_settings)

    component_extended = gf.c.extend_ports(
        component, length=ss.distance_source_to_monitors
    )

    ports = component_extended.get_ports_list(port_type="optical")
    if not ports:
        raise ValueError(f"`{component.name}` does not have any optical ports")

    c = gf.components.extension.extend_ports(
        component=component, length=ss.port_extension
    )
    c.remove_layers(component.layers - set(layer_to_thickness.keys()))
    c._bb_valid = False
    c.flatten()
    c.name = "top"
    gdspath = c.write_gds()

    filepath = get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_to_material,
        layer_to_thickness=layer_to_thickness,
        **settings,
    )
    filepath_csv = filepath.with_suffix(".csv")
    filepath_sim_settings = filepath.with_suffix(".yml")
    filepath_fsp = filepath.with_suffix(".fsp")

    if run and filepath_csv.exists() and not overwrite:
        logger.info(f"Reading Sparameters from {filepath_csv}")
        return pd.read_csv(filepath_csv)

    if not run and session is None:
        print(run_false_warning)

    logger.info(f"Writing Sparameters to {filepath_csv}")
    x_min = (component.xmin - ss.xmargin) * 1e-6
    x_max = (component.xmax + ss.xmargin) * 1e-6
    y_min = (component.ymin - ss.ymargin) * 1e-6
    y_max = (component.ymax + ss.ymargin) * 1e-6

    layers_thickness = [
        layer_to_thickness[layer]
        for layer in component.get_layers()
        if layer in layer_to_thickness
    ]
    if not layers_thickness:
        raise ValueError(
            f"no layers for component {component.get_layers()}"
            f"in layer stack {layers_thickness.keys()}"
        )
    layers_zmin = [
        layer_to_zmin[layer]
        for layer in component.get_layers()
        if layer in layer_to_zmin
    ]
    component_thickness = max(layers_thickness)
    component_zmin = min(layers_zmin)

    z = (component_zmin + component_thickness) / 2 * 1e-6
    z_span = (2 * ss.zmargin + component_thickness) * 1e-6

    x_span = x_max - x_min
    y_span = y_max - y_min

    layers = c.get_layers()
    sim_settings.update(dict(layer_stack=layer_stack.to_dict()))

    sim_settings = dict(
        simulation_settings=sim_settings,
        component=component.to_dict(),
        version=__version__,
    )

    logger.info(
        f"Simulation size = {x_span*1e6:.3f}, {y_span*1e6:.3f}, {z_span*1e6:.3f} um"
    )

    # from pprint import pprint
    # filepath_sim_settings.write_text(omegaconf.OmegaConf.to_yaml(sim_settings))
    # print(filepath_sim_settings)
    # pprint(sim_settings)
    # return

    try:
        import lumapi
    except ModuleNotFoundError as e:
        print(
            "Cannot import lumapi (Python Lumerical API). "
            "You can add set the PYTHONPATH variable or add it with `sys.path.append()`"
        )
        raise e
    except OSError as e:
        raise e

    start = time.time()
    s = session or lumapi.FDTD(hide=False)
    s.newproject()
    s.selectall()
    s.deleteall()
    s.addrect(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z=z,
        z_span=z_span,
        index=1.5,
        name="clad",
    )

    material = ss.background_material
    if material not in MATERIAL_NAME_TO_LUMERICAL:
        raise ValueError(
            f"{material!r} not in {list(MATERIAL_NAME_TO_LUMERICAL.keys())}"
        )
    material = MATERIAL_NAME_TO_LUMERICAL[material]
    s.setnamed("clad", "material", material)

    s.addfdtd(
        dimension="3D",
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z=z,
        z_span=z_span,
        mesh_accuracy=ss.mesh_accuracy,
        use_early_shutoff=True,
        simulation_time=ss.simulation_time,
        simulation_temperature=ss.simulation_temperature,
    )

    for layer, thickness in layer_to_thickness.items():
        if layer not in layers:
            continue

        if layer not in layer_to_material:
            raise ValueError(f"{layer!r} not in {layer_to_material.keys()}")

        material_name = layer_to_material[layer]
        if material_name not in MATERIAL_NAME_TO_LUMERICAL:
            raise ValueError(
                f"{material_name!r} not in {list(MATERIAL_NAME_TO_LUMERICAL.keys())}"
            )
        material_name_lumerical = MATERIAL_NAME_TO_LUMERICAL[material_name]

        if layer not in layer_to_zmin:
            raise ValueError(f"{layer} not in {list(layer_to_zmin.keys())}")

        zmin = layer_to_zmin[layer]
        zmax = zmin + thickness
        z = (zmax + zmin) / 2

        s.gdsimport(str(gdspath), "top", f"{layer[0]}:{layer[1]}")
        layername = f"GDS_LAYER_{layer[0]}:{layer[1]}"
        s.setnamed(layername, "z", z * 1e-6)
        s.setnamed(layername, "z span", thickness * 1e-6)
        s.setnamed(layername, "material", material_name_lumerical)
        logger.info(f"adding {layer}, thickness = {thickness} um, zmin = {zmin} um ")

    for i, port in enumerate(ports):
        zmin = layer_to_zmin[port.layer]
        thickness = layer_to_thickness[port.layer]
        z = (zmin + thickness) / 2
        zspan = 2 * ss.port_margin + thickness

        s.addport()
        p = f"FDTD::ports::port {i+1}"
        s.setnamed(p, "x", port.x * 1e-6)
        s.setnamed(p, "y", port.y * 1e-6)
        s.setnamed(p, "z", z * 1e-6)
        s.setnamed(p, "z span", zspan * 1e-6)
        s.setnamed(p, "frequency dependent profile", ss.frequency_dependendent_profile)
        s.setnamed(p, "number of field profile samples", ss.field_profile_samples)

        deg = int(port.orientation)
        # if port.orientation not in [0, 90, 180, 270]:
        #     raise ValueError(f"{port.orientation} needs to be [0, 90, 180, 270]")

        if -45 <= deg <= 45:
            direction = "Backward"
            injection_axis = "x-axis"
            dxp = 0
            dyp = 2 * ss.port_margin + port.width
        elif 45 < deg < 90 + 45:
            direction = "Backward"
            injection_axis = "y-axis"
            dxp = 2 * ss.port_margin + port.width
            dyp = 0
        elif 90 + 45 < deg < 180 + 45:
            direction = "Forward"
            injection_axis = "x-axis"
            dxp = 0
            dyp = 2 * ss.port_margin + port.width
        elif 180 + 45 < deg < 180 + 45 + 90:
            direction = "Forward"
            injection_axis = "y-axis"
            dxp = 2 * ss.port_margin + port.width
            dyp = 0

        else:
            raise ValueError(
                f"port {port.name} orientation {port.orientation} is not valid"
            )

        s.setnamed(p, "direction", direction)
        s.setnamed(p, "injection axis", injection_axis)
        s.setnamed(p, "y span", dyp * 1e-6)
        s.setnamed(p, "x span", dxp * 1e-6)
        # s.setnamed(p, "theta", deg)
        s.setnamed(p, "name", port.name)
        # s.setnamed(p, "name", f"o{i+1}")

        logger.info(
            f"port {p} {port.name}: at ({port.x}, {port.y}, 0)"
            f"size = ({dxp}, {dyp}, {zspan})"
        )

    s.setglobalsource("wavelength start", ss.wavelength_start * 1e-6)
    s.setglobalsource("wavelength stop", ss.wavelength_stop * 1e-6)
    s.setnamed("FDTD::ports", "monitor frequency points", ss.wavelength_points)

    if run:
        s.save(str(filepath_fsp))
        s.deletesweep("s-parameter sweep")

        s.addsweep(3)
        s.setsweep("s-parameter sweep", "Excite all ports", 0)
        s.setsweep("S sweep", "auto symmetry", True)
        s.runsweep("s-parameter sweep")
        sp = s.getsweepresult("s-parameter sweep", "S parameters")
        s.exportsweep("s-parameter sweep", str(filepath))
        logger.info(f"wrote sparameters to {filepath}")

        keys = [key for key in sp.keys() if key.startswith("S")]
        ra = {f"{key}a": list(np.unwrap(np.angle(sp[key].flatten()))) for key in keys}
        rm = {f"{key}m": list(np.abs(sp[key].flatten())) for key in keys}
        wavelengths = sp["lambda"].flatten() * 1e6

        results = {"wavelengths": wavelengths}
        results.update(ra)
        results.update(rm)
        df = pd.DataFrame(results, index=wavelengths)

        end = time.time()
        df.to_csv(filepath_csv, index=False)
        sim_settings.update(compute_time_seconds=end - start)
        filepath_sim_settings.write_text(omegaconf.OmegaConf.to_yaml(sim_settings))
        return df
    filepath_sim_settings.write_text(omegaconf.OmegaConf.to_yaml(sim_settings))
    return s


def _sample_write_coupler_ring():
    """Write Sparameters when changing a component setting."""
    return [
        write_sparameters_lumerical(
            gf.components.coupler_ring(
                width=width, length_x=length_x, radius=radius, gap=gap
            )
        )
        for width in [0.5]
        for length_x in [0.1, 1, 2, 3, 4]
        for gap in [0.15, 0.2]
        for radius in [5, 10]
    ]


def _sample_bend_circular():
    """Write Sparameters for a circular bend with different radius."""
    return [
        write_sparameters_lumerical(gf.components.bend_circular(radius=radius))
        for radius in [2, 5, 10]
    ]


def _sample_bend_euler():
    """Write Sparameters for a euler bend with different radius."""
    return [
        write_sparameters_lumerical(gf.components.bend_euler(radius=radius))
        for radius in [2, 5, 10]
    ]


def _sample_convergence_mesh():
    return [
        write_sparameters_lumerical(
            component=gf.components.straight(length=2),
            mesh_accuracy=mesh_accuracy,
        )
        for mesh_accuracy in [1, 2, 3]
    ]


def _sample_convergence_wavelength():
    return [
        write_sparameters_lumerical(
            component=gf.components.straight(length=2),
            wavelength_start=wavelength_start,
        )
        for wavelength_start in [1.2, 1.4]
    ]


if __name__ == "__main__":
    # component = gf.components.straight(length=2.5)
    component = gf.components.mmi1x2()
    r = write_sparameters_lumerical(
        component=component, mesh_accuracy=1, wavelength_points=200, run=False
    )
    # c = gf.components.coupler_ring(length_x=3)
    # c = gf.components.mmi1x2()
    # print(r)
    # print(r.keys())
    # print(component.ports.keys())
