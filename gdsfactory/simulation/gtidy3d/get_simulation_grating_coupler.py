"""Returns tidy3d simulation from gdsfactory Component."""
import warnings
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pydantic
import tidy3d as td

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.extension import move_polar_rad_copy
from gdsfactory.config import logger
from gdsfactory.routing.sort_ports import sort_ports_x, sort_ports_y
from gdsfactory.simulation.gtidy3d.materials import get_index, get_medium
from gdsfactory.tech import LAYER_STACK, LayerStack

MATERIAL_NAME_TO_TIDY3D = {
    # "si": 3.47,
    # "sio2": 1.44,
    # "sin": 2.0,
    "si": "cSi",
    "sio2": "SiO2",
    "sin": "Si3N4",
}


@pydantic.validate_arguments
def get_simulation_grating_coupler(
    component: Component,
    port_extension: Optional[float] = 4.0,
    layer_stack: LayerStack = LAYER_STACK,
    thickness_pml: float = 1.0,
    xmargin: float = 0,
    ymargin: float = 0,
    xmargin_left: float = 0,
    xmargin_right: float = 0,
    ymargin_top: float = 0,
    ymargin_bot: float = 0,
    zmargin: float = 1.0,
    clad_material: str = "sio2",
    port_source_name: str = "o1",
    port_margin: float = 0.5,
    port_source_offset: float = 0.1,
    distance_source_to_monitors: float = 0.2,
    resolution: float = 50,
    wavelength_start: float = 1.50,
    wavelength_stop: float = 1.60,
    wavelength_points: int = 50,
    plot_modes: bool = False,
    num_modes: int = 2,
    run_time_ps: float = 10.0,
    fiber_port_type: str = "vertical_te",
    fiber_xoffset: float = 0,
    fiber_z: float = 2,
    fiber_mfd: float = 5.2,
    fiber_angle_deg: float = 20.0,
    material_name_to_tidy3d: Dict[str, Union[float, str]] = MATERIAL_NAME_TO_TIDY3D,
) -> td.Simulation:
    r"""Returns Simulation object from gdsfactory.component

    injects a Gaussian beam from above and monitors the transmission into the waveguide.

    based on grating coupler example
    https://docs.simulation.cloud/projects/tidy3d/en/latest/notebooks/GratingCoupler.html

    .. code::

         top view
              ________________________________
             |                               |
             | xmargin_left                  |
             |<------>                       |
             |           ________________    |
             |          /   |  |  |  |  |    |
             |         /    |  |  |  |  |    |
             |=========     |  |  |  |  |    |
             |         \    |  |  |  |  |    |
             |   _ _ _ _\___|__|__|__|__| ___|
             |   |                       <-->|
             |   |ymargin_bot   xmargin_right|
             |   |                           |
             |___|___________________________|

        side view
                     waist_radius
                 /     /  /     /       |
                /     /  /     /        | fiber_thickness
               /     /  /     /    _ _ _| _ _ _ _ _ _  _
                                        |
                                        | air_gap_thickness
                                   _ _ _| _ _ _ _ _ _  _
                                        |
                       nclad            | top_clad_thickness
                    _   _   _      _ _ _| _ _ _ _ _ _  _
              nwg _| |_| |_| |__________|              _
                                        |               |
                     nslab              |wg_thickness   | slab_thickness
                    ______________ _ _ _|_ _ _ _ _ _ _ _|
                                        |
                     nbox               |box_thickness
                    ______________ _ _ _|_ _ _ _ _ _ _ _
                                        |
                     nsubstrate         |substrate_thickness
                    ______________ _ _ _|

        |--------------------|<-------->
                                xmargin

    Args:
        component: gdsfactory Component.
        port_extension: extend ports beyond the PML.
        layer_stack: contains layer numbers (int, int) to thickness, zmin.
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
        port_source_offset: mode solver workaround.
            positive moves source forward, negative moves source backward.
        resolution: in pixels/um (20: for coarse, 120: for fine)
        wavelength_start: in (um).
        wavelength_stop: in (um).
        wavelength_points: number of wavelengths.
        plot_modes: plot source modes.
        num_modes: number of modes to plot.
        run_time_ps: make sure it's sufficient for the fields to decay.
            defaults to 10ps and counts on the automatic shutoff to stop earlier if needed.


    .. code::

        import matplotlib.pyplot as plt
        import gdsfactory as gf
        import gdsfactory.simulation.tidy3d as gt

        c = gf.components.bend_circular()
        sim = gt.get_simulation(c)
        gt.plot_simulation(sim)

    """
    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_material = layer_stack.get_layer_to_material()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    # layer_to_sidewall_angle = layer_stack.get_layer_to_sidewall_angle()

    assert isinstance(
        component, Component
    ), f"component needs to be a gf.Component, got Type {type(component)}"
    if port_source_name not in component.ports:
        warnings.warn(
            f"port_source_name={port_source_name} not in {component.ports.keys()}"
        )
        port_source = component.get_ports_list()[0]
        port_source_name = port_source.name
        warnings.warn(f"Selecting port_source_name={port_source_name} instead.")

    component_padding = gf.add_padding_container(
        component,
        default=0,
        top=ymargin or ymargin_top,
        bottom=ymargin or ymargin_bot,
        left=xmargin or xmargin_left,
        right=xmargin or xmargin_right,
    )
    component_extended = (
        gf.components.extension.extend_ports(
            component=component_padding, length=port_extension, centered=True
        )
        if port_extension
        else component
    )

    gf.show(component_extended)
    component_extended.flatten()

    component_ref = component_padding.ref()
    component_ref.x = 0
    component_ref.y = 0

    clad_material_name_or_index = material_name_to_tidy3d[clad_material]
    clad = td.Structure(
        geometry=td.Box(
            size=(td.inf, td.inf, td.inf),
            center=(0, 0, 0),
        ),
        medium=get_medium(name_or_index=clad_material_name_or_index),
    )
    structures = [clad]

    layers_thickness = [
        layer_to_thickness[layer]
        for layer in component.get_layers()
        if layer in layer_to_thickness
    ]

    t_core = max(layers_thickness)
    cell_thickness = thickness_pml + t_core + thickness_pml + 2 * zmargin
    sim_size = [
        component_ref.xsize + 2 * thickness_pml,
        component_ref.ysize + 2 * thickness_pml,
        cell_thickness,
    ]

    for layer in component.layers:
        if layer in layer_to_thickness and layer in layer_to_material:
            thickness = layer_to_thickness[layer]
            zmin = layer_to_zmin[layer]
            zmax = zmin + thickness
            if (
                layer in layer_to_material
                and layer_to_material[layer] in material_name_to_tidy3d
            ):
                name_or_index = material_name_to_tidy3d[layer_to_material[layer]]
                medium = get_medium(name_or_index=name_or_index)
                index = get_index(name_or_index=name_or_index)
                logger.debug(
                    f"Add {layer}, {name_or_index!r}, index = {index:.3f}, "
                    f"thickness = {thickness}, zmin = {zmin}, zmax = {zmax}"
                )

                polygons = td.PolySlab.from_gds(
                    gds_cell=component_extended,
                    gds_layer=layer[0],
                    gds_dtype=layer[1],
                    axis=2,
                    slab_bounds=(zmin, zmax),
                )

                for polygon in polygons:
                    geometry = td.Structure(
                        geometry=polygon,
                        medium=medium,
                    )
                    structures.append(geometry)
            elif layer not in layer_to_material:
                logger.debug(f"Layer {layer} not in {layer_to_material.keys()}")
            elif layer_to_material[layer] not in material_name_to_tidy3d:
                materials = list(material_name_to_tidy3d.keys())
                logger.debug(f"material {layer_to_material[layer]} not in {materials}")

    wavelengths = np.linspace(wavelength_start, wavelength_stop, wavelength_points)
    freqs = td.constants.C_0 / wavelengths
    freq0 = td.constants.C_0 / np.mean(wavelengths)
    fwidth = freq0 / 10

    # Add input waveguide source
    port = component_ref.ports[port_source_name]
    angle = port.orientation
    width = port.width + 2 * port_margin
    size_x = width * abs(np.sin(angle * np.pi / 180))
    size_y = width * abs(np.cos(angle * np.pi / 180))
    size_x = 0 if size_x < 0.001 else size_x
    size_y = 0 if size_y < 0.001 else size_y
    size_z = cell_thickness - 2 * zmargin

    source_size = [size_x, size_y, size_z]
    xy_shifted = move_polar_rad_copy(
        np.array(port.center), angle=angle * np.pi / 180, length=port_source_offset
    )
    source_center_offset = xy_shifted.tolist() + [0]  # (x, y, z=0)

    # Add waveguide port monitor
    ports = sort_ports_x(
        sort_ports_y(component_ref.get_ports_list(port_type="optical"))
    )
    ports = component_ref.get_ports_list(port_type=fiber_port_type)
    assert len(ports) == 1, f"More than one optical found {ports}"
    port = ports[0]

    port_name = port.name
    angle = port.orientation
    width = port.width + 2 * port_margin
    size_x = width * abs(np.sin(angle * np.pi / 180))
    size_y = width * abs(np.cos(angle * np.pi / 180))
    size_x = 0 if size_x < 0.001 else size_x
    size_y = 0 if size_y < 0.001 else size_y
    size = (size_x, size_y, size_z)

    # if monitor has a source move monitor inwards
    length = -distance_source_to_monitors if port_name == port_source_name else 0
    xy_shifted = move_polar_rad_copy(
        np.array(port.center), angle=angle * np.pi / 180, length=length
    )
    center = xy_shifted.tolist() + [0]  # (x, y, z=0)

    flux_monitor = td.ModeMonitor(
        center=center,
        size=size,
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=1),
        name="flux",
    )

    # Add fiber monitor
    ports = component_ref.get_ports_list(port_type=fiber_port_type)
    assert len(ports) == 1, f"More than one port_type={fiber_port_type!r} found {ports}"
    fiber_port = ports[0]

    # inject Gaussian beam from above and monitors the transmission into the waveguide.
    gaussian_beam = td.GaussianBeam(
        size=(td.inf, td.inf, 0),
        center=[fiber_port.x + fiber_xoffset, 0, fiber_z],
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        angle_theta=np.deg2rad(fiber_angle_deg),
        angle_phi=np.pi,
        direction="-",
        waist_radius=fiber_mfd / 2,
        pol_angle=np.pi / 2,
    )

    plane_monitor = td.FieldMonitor(
        center=[0, 0, (zmax + zmin) / 2],
        size=[sim_size[0], sim_size[1], 0],
        freqs=[freq0],
        name="full_domain_fields",
    )

    rad_monitor = td.FieldMonitor(
        center=[0, 0, 0],
        size=[td.inf, 0, td.inf],
        freqs=[freq0],
        name="radiated_fields",
    )

    near_field_monitor = td.FieldMonitor(
        center=[0, 0, fiber_z],
        size=[td.inf, td.inf, 0],
        freqs=[freq0],
        name="radiated_near_fields",
    )

    sim = td.Simulation(
        size=sim_size,
        grid_size=3 * [1 / resolution],
        structures=structures,
        sources=[gaussian_beam],
        monitors=[plane_monitor, rad_monitor, flux_monitor, near_field_monitor],
        run_time=20 * run_time_ps / fwidth,
        pml_layers=3 * [td.PML()],
    )

    if plot_modes:
        src_plane = td.Box(center=source_center_offset, size=source_size)
        ms = td.plugins.ModeSolver(simulation=sim, plane=src_plane, freq=freq0)

        mode_spec = td.ModeSpec(num_modes=num_modes)
        modes = ms.solve(mode_spec=mode_spec)

        print(
            "Effective index of computed modes: ",
            ", ".join([f"{mode.n_eff:1.4f}" for mode in modes]),
        )

        fig, axs = plt.subplots(num_modes, 2, figsize=(12, 12))
        for mode_ind in range(num_modes):
            abs(modes[mode_ind].field_data.Ey).plot(
                x="y", y="z", cmap="magma", ax=axs[mode_ind, 0]
            )
            abs(modes[mode_ind].field_data.Ez).plot(
                x="y", y="z", cmap="magma", ax=axs[mode_ind, 1]
            )
            axs[mode_ind, 0].set_aspect("equal")
            axs[mode_ind, 1].set_aspect("equal")
        plt.show()
    return sim


if __name__ == "__main__":
    import gdsfactory.simulation.gtidy3d as gt

    c = gf.components.grating_coupler_elliptical_lumerical()
    sim = get_simulation_grating_coupler(c, plot_modes=False)

    # gt.plot_simulation(sim) # make sure simulations looks good

    sim_data = gt.get_results(sim).result()
    freq0 = td.constants.C_0 / 1.55
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, tight_layout=True, figsize=(14, 16))
    sim_data.plot_field("full_domain_fields", "Ey", freq=freq0, z=0, ax=ax1)
    sim_data.plot_field("radiated_near_fields", "Ey", freq=freq0, z=0, ax=ax2)
    sim_data.plot_field("radiated_fields", "Ey", freq=freq0, y=0, ax=ax3)
