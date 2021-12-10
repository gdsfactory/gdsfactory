"""Returns tidy3d simulation from gdsfactory Component."""
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pydantic
import tidy3d as td

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.extension import move_polar_rad_copy
from gdsfactory.routing.sort_ports import sort_ports_x, sort_ports_y
from gdsfactory.simulation.tidy3d.materials import get_material
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
    zmargin: float = 1.0,
    thickness_pml: float = 1.0,
    clad_material: str = "SiO2",
    port_source_name: str = "o1",
    port_margin: float = 0.5,
    distance_source_to_monitors: float = 0.2,
    mesh_step: float = 40e-3,
    wavelength: float = 1.55,
) -> td.Simulation:
    """Returns Simulation object from gdsfactory.component

    based on GDS example
    https://simulation.cloud/docs/html/examples/ParameterScan.html

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
        mesh_step: in all directions
        wavelength: in (um)

    You can visualize the simulation with gdsfactory


    .. code::

        import matplotlib.pyplot as plt
        import gdsfactory as gf
        import gdsfactory.simulation.tidy3d as gm

        c = gf.components.bend_circular()
        sim = gm.get_simulation(c)
        gm.plot_simulation(sim)

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

    component_extended = (
        gf.components.extension.extend_ports(
            component=component, length=port_extension, centered=True
        )
        if port_extension
        else component
    )

    gf.show(component_extended)
    component_extended.flatten()
    component_extended_ref = component_extended.ref()

    component_ref = component.ref()
    component_ref.x = 0
    component_ref.y = 0

    structures = [
        td.Box(
            material=get_material(name=clad_material),
            size=(td.inf, td.inf, td.inf),
            center=(0, 0, 0),
        )
    ]
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
            height = layer_to_thickness[layer]
            zmin = layer_to_zmin[layer]
            z_cent = zmin + height / 2
            material_name = MATERIAL_NAME_TO_TIDY3D[layer_to_material[layer]]
            material = get_material(name=material_name)

            geometry = td.GdsSlab(
                material=material,
                gds_cell=component_extended_ref,
                gds_layer=layer[0],
                gds_dtype=layer[1],
                z_cent=z_cent,
                z_size=height,
            )
            structures.append(geometry)

    # Add source
    port = component_ref.ports[port_source_name]
    angle = port.orientation
    width = port.width + 2 * port_margin
    size_x = width * abs(np.sin(angle * np.pi / 180))
    size_y = width * abs(np.cos(angle * np.pi / 180))
    size_x = 0 if size_x < 0.001 else size_x
    size_y = 0 if size_y < 0.001 else size_y
    size_z = cell_thickness - 2 * thickness_pml
    size = [size_x, size_y, size_z]
    center = port.center.tolist() + [0]  # (x, y, z=0)
    freq0 = td.constants.C_0 / wavelength
    fwidth = freq0 / 10

    msource = td.ModeSource(
        size=size,
        center=center,
        source_time=td.GaussianPulse(frequency=freq0, fwidth=fwidth),
        direction="forward",
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
            center=[port.x, port.y, t_core / 2],
            size=size,
            freqs=[freq0],
            Nmodes=1,
            name=port.name,
        )

    domain_monitor = td.FreqMonitor(
        center=[0, 0, z_cent], size=[sim_size[0], sim_size[1], 0], freqs=[freq0]
    )

    sim = td.Simulation(
        size=sim_size,
        mesh_step=mesh_step,
        structures=structures,
        sources=[msource],
        monitors=[domain_monitor] + list(monitors.values()),
        run_time=20 / fwidth,
        pml_layers=[12, 12, 12],
    )
    # set the modes
    sim.compute_modes(msource, Nmodes=n_modes)
    sim.set_mode(msource, mode_ind=mode_index)
    return sim


def plot_simulation(
    sim: td.Simulation,
    normal1: str = "z",
    normal2: str = "x",
    position1: float = 0.0,
    position2: float = 0.0,
):
    """Returns figure with two axis of the Simulation.

    Args:
        sim: simulation object
        normal1: {'x', 'y', 'z'} Axis normal to the cross-section plane.
        normal2: {'x', 'y', 'z'} Axis normal to the cross-section plane.
        position1: Position offset along the normal axis.
        position2: Position offset along the normal axis.

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    sim.viz_eps_2D(normal=normal1, position=position1, ax=ax1)
    sim.viz_eps_2D(normal=normal2, position=position2, ax=ax2, source_alpha=1)
    plt.show()
    return fig


def plot_materials(
    sim: td.Simulation,
    normal1: str = "z",
    normal2: str = "x",
    position1: float = 0.0,
    position2: float = 0.0,
):
    """Returns figure with two axis of the Simulation.

    Args:
        sim: simulation object
        normal1: {'x', 'y', 'z'} Axis normal to the cross-section plane.
        normal2: {'x', 'y', 'z'} Axis normal to the cross-section plane.
        position1: Position offset along the normal axis.
        position2: Position offset along the normal axis.

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    sim.viz_mat_2D(normal=normal1, position=position1, ax=ax1)
    sim.viz_mat_2D(
        normal=normal2, position=position2, ax=ax2, source_alpha=1, legend=True
    )
    plt.show()
    return fig


if __name__ == "__main__":
    c = gf.components.mmi1x2()
    c = gf.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    # c = gf.add_padding(c, default=0, bottom=2, right=2, layers=[(100, 0)])

    # c = gf.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])
    # c = gf.components.straight(length=2)

    c = gf.components.bend_circular(radius=2)

    sim = get_simulation(c)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    # sim.viz_eps_2D(normal="z", position=0, ax=ax1)
    # sim.viz_eps_2D(normal="x", ax=ax2, source_alpha=1)
    # ax2.set_xlim([-3, 3])
    plot_simulation(sim)
