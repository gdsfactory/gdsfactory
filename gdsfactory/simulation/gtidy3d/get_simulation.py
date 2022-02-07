"""Returns tidy3d simulation from gdsfactory Component."""
import warnings
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydantic
import tidy3d as td

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.extension import move_polar_rad_copy
from gdsfactory.config import logger
from gdsfactory.routing.sort_ports import sort_ports_x, sort_ports_y
from gdsfactory.simulation.gtidy3d.materials import get_medium
from gdsfactory.tech import LAYER_STACK, LayerStack

MATERIAL_NAME_TO_TIDY3D = {
    "si": "cSi",
    "sio2": "SiO2",
    "sin": "Si3N4",
}


@pydantic.validate_arguments
def get_simulation(
    component: Component,
    mode_index: int = 0,
    n_modes: int = 2,
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
    distance_source_to_monitors: float = 0.2,
    resolution: float = 50,
    wavelength: float = 1.55,
    plot_modes: bool = False,
) -> td.Simulation:
    r"""Returns Simulation object from gdsfactory.component

    based on GDS example
    https://simulation.cloud/docs/html/examples/ParameterScan.html

    .. code::

         top view
              ________________________________
             |                               |
             | xmargin_left                  | port_extension
             |<------>          port_margin ||<-->
          ___|___________          _________||___
             |           \        /          |
             |            \      /           |
             |             ======            |
             |            /      \           |
          ___|___________/        \__________|___
             |   |                 <-------->|
             |   |ymargin_bot   xmargin_right|
             |   |                           |
             |___|___________________________|

        side view
              ________________________________
             |                     |         |
             |                     |         |
             |                   zmargin_top |
             |xmargin_left         |         |
             |<---> _____         _|___      |
             |     |     |       |     |     |
             |     |     |       |     |     |
             |     |_____|       |_____|     |
             |       |                       |
             |       |                       |
             |       |zmargin_bot            |
             |       |                       |
             |_______|_______________________|


    Args:
        component: gf.Component
        mode_index: mode index
        n_modes: number of modes
        port_extension: extend ports beyond the PML
        layer_stack: contains layer numbers (int, int) to thickness, zmin
        zmargin: thickness for cladding above and below core
        thickness_pml: PML thickness (um)
        clad_material: material for cladding
        port_source_name: input port name
        port_margin: margin on each side of the port
        distance_source_to_monitors: in (um) source goes before monitors
        resolution: grid_size=3*[1/resolution]
        wavelength: in (um)
        plot_modes: plot source modes.
        xmargin: left and right distance from component to PML.
        xmargin_left: west distance from component to PML.
        xmargin_right: east distance from component to PML.
        ymargin: top and bottom distance from component to PML.
        ymargin_top: north distance from component to PML.
        ymargin_bot: south distance from component to PML.


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
        top=ymargin_top,
        bottom=ymargin_bot,
        left=xmargin_left,
        right=xmargin_right,
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

    clad = td.Structure(
        geometry=td.Box(
            size=(td.inf, td.inf, td.inf),
            center=(0, 0, 0),
        ),
        medium=get_medium(name=clad_material),
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

    layer_to_polygons = component_extended.get_polygons(by_spec=True)

    for layer in component.layers:
        if layer in layer_to_thickness and layer in layer_to_material:
            thickness = layer_to_thickness[layer]
            zmin = layer_to_zmin[layer]
            zmax = zmin + thickness
            if (
                layer in layer_to_material
                and layer_to_material[layer] in MATERIAL_NAME_TO_TIDY3D
            ):
                material_name = MATERIAL_NAME_TO_TIDY3D[layer_to_material[layer]]
                medium = get_medium(name=material_name)
                logger.debug(
                    f"Add {layer}, thickness = {thickness}, zmin = {zmin}, zmax = {zmax}"
                )

                npolygons = len(layer_to_polygons[layer])
                for polygon_index in range(npolygons):
                    poly = td.PolySlab.from_gds(
                        gds_cell=component_extended,
                        gds_layer=layer[0],
                        gds_dtype=layer[1],
                        axis=2,
                        slab_bounds=(zmin, zmax),
                        polygon_index=polygon_index,
                    )
                    geometry = td.Structure(
                        geometry=poly,
                        medium=medium,
                    )
                    structures.append(geometry)
            elif layer not in layer_to_material:
                logger.debug(f"Layer {layer} not in {layer_to_material.keys()}")
            elif layer_to_material[layer] not in MATERIAL_NAME_TO_TIDY3D:
                logger.debug(
                    f"material {layer_to_material[layer]} not in {MATERIAL_NAME_TO_TIDY3D.keys()}"
                )
    # Add source
    port = component_ref.ports[port_source_name]
    angle = port.orientation
    width = port.width + 2 * port_margin
    size_x = width * abs(np.sin(angle * np.pi / 180))
    size_y = width * abs(np.cos(angle * np.pi / 180))
    size_x = 0 if size_x < 0.001 else size_x
    size_y = 0 if size_y < 0.001 else size_y
    size_z = cell_thickness - 2 * zmargin

    source_size = [size_x, size_y, size_z]
    source_center = port.center.tolist() + [0]  # (x, y, z=0)
    freq0 = td.constants.C_0 / wavelength
    fwidth = freq0 / 10

    msource = td.ModeSource(
        size=source_size,
        center=source_center,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="+",
    )

    # Add port monitors
    monitors = {}
    ports = sort_ports_x(sort_ports_y(component_ref.get_ports_list()))
    for port in ports:
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

        monitors[port_name] = td.ModeMonitor(
            center=center,
            size=size,
            freqs=[freq0],
            mode_spec=td.ModeSpec(num_modes=1),
            name=port.name,
        )

    domain_monitor = td.FieldMonitor(
        center=[0, 0, (zmax + zmin) / 2],
        size=[sim_size[0], sim_size[1], 0],
        freqs=[freq0],
        name="field",
    )

    sim = td.Simulation(
        size=sim_size,
        grid_size=3 * [1 / resolution],
        structures=structures,
        sources=[msource],
        monitors=[domain_monitor] + list(monitors.values()),
        run_time=20 / fwidth,
        pml_layers=3 * [td.PML()],
    )

    if plot_modes:
        src_plane = td.Box(center=source_center, size=source_size)
        ms = td.plugins.ModeSolver(simulation=sim, plane=src_plane, freq=freq0)
        mode_spec = td.ModeSpec(num_modes=3)
        modes = ms.solve(mode_spec=mode_spec)

        print(
            "Effective index of computed modes: ",
            ", ".join([f"{mode.n_eff:1.4f}" for mode in modes]),
        )

        fig, axs = plt.subplots(3, 2, figsize=(12, 12))
        for mode_ind in range(3):
            abs(modes[mode_ind].field_data.Ey).plot(
                x="y", y="z", cmap="magma", ax=axs[mode_ind, 0]
            )
            abs(modes[mode_ind].field_data.Ez).plot(
                x="y", y="z", cmap="magma", ax=axs[mode_ind, 1]
            )
            axs[mode_ind, 0].set_aspect("equal")
            axs[mode_ind, 1].set_aspect("equal")
        # plt.show()
    return sim


def plot_simulation_yz(sim: td.Simulation, z: float = 0.0, y: float = 0.0):
    """Returns figure with two axis of the Simulation.

    Args:
        sim: simulation object
        z:
        y:
    """

    fig = plt.figure(figsize=(11, 4))
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    sim.plot(z=z, ax=ax1)
    sim.plot(y=y, ax=ax2)
    return fig


def plot_simulation_xz(sim: td.Simulation, x: float = 0.0, z: float = 0.0):
    """Returns figure with two axis of the Simulation.

    Args:
        sim: simulation object
        x:
        z:
    """

    fig = plt.figure(figsize=(11, 4))
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    sim.plot(z=z, ax=ax1)
    sim.plot(x=x, ax=ax2)
    return fig


plot_simulation = plot_simulation_yz


if __name__ == "__main__":
    # c = gf.components.mmi1x2()
    # c = gf.components.bend_circular(radius=2)
    # c = gf.components.crossing()
    c = gf.c.straight_rib()

    plot_modes = False
    plot_modes = True
    sim = get_simulation(c, plot_modes=plot_modes)
    # plot_simulation(sim)

    fig = plt.figure(figsize=(11, 4))
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    sim.plot(z=0.0, ax=ax1)
    sim.plot(x=0.0, ax=ax2)
    plt.show()
