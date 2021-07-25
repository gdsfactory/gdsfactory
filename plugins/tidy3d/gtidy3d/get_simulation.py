"""
Returns simulation from component

Questions:

- How to define a field_monitor_port
    field_monitor_port = component.ports[port_field_monitor_name]
    field_monitor_point = field_monitor_port.center.tolist() + [0]  # (x, y, z=0)

"""
from typing import Dict, Optional, Tuple


import matplotlib.pyplot as plt
import numpy as np
import pp
from numpy import ndarray
from pp.component import Component
from pp.components.extension import move_polar_rad_copy

import tidy3d as td
from tidy3d import web


LAYER_TO_THICKNESS_NM = {(1, 0): 220.0}
LAYER_TO_MATERIAL = {(1, 0): "Si"}
LAYER_TO_ZMIN = {(1, 0): "Si"}
LAYER_TO_SIDEWALL_ANGLE = {(1, 0): "Si"}


# wavelength / frequency
lambda0 = 1.550  # all length scales in microns
freq0 = td.constants.C_0 / lambda0
fwidth = freq0 / 10

# Permittivity of waveguide and substrate
wg_n = 3.48
sub_n = 1.45
mat_wg = td.Medium(n=wg_n)
mat_sub = td.Medium(n=sub_n)

# Waveguide dimensions

# Waveguide height
wg_height = 0.22
# Waveguide width
wg_width = 0.45
# Waveguide separation in the beginning/end
wg_spacing_in = 8
# Total device length along propagation direction
device_length = 100
# Length of the bend region
bend_length = 16
# space between waveguide and PML
pml_spacing = 1
# Mesh step in all directions
mesh_step = 0.040


def get_simulation(
    component: Component,
    extend_ports_length: Optional[float] = 4.0,
    layer_to_thickness_nm: Dict[Tuple[int, int], float] = LAYER_TO_THICKNESS_NM,
    layer_to_material: Dict[Tuple[int, int], str] = LAYER_TO_MATERIAL,
    layer_to_zmin_nm: Dict[Tuple[int, int], float] = {(1, 0): 0.5},
    layer_to_sidewall_angle: Dict[Tuple[int, int], float] = {(1, 0): 0},
    res: int = 20,
    t_clad_top: float = 1.0,
    t_clad_bot: float = 1.0,
    tpml: float = 1.0,
    clad_material: str = "SiO2",
    is_3d: bool = False,
    wavelengths: ndarray = np.linspace(1.5, 1.6, 50),
    dfcen: float = 0.2,
    port_source_name: str = "W0",
    port_field_monitor_name: str = "E0",
    port_margin: float = 0.5,
    distance_source_to_monitors: float = 0.2,
) -> Tuple[td.Simulation, int]:
    """Returns Simulation dict from gdsfactory component

    based on GDS example
    https://simulation.cloud/docs/html/examples/ParameterScan.html

    https://support.lumerical.com/hc/en-us/articles/360042095873-Metamaterial-S-parameter-extraction

    Args:
        component: gdsfactory Component
        extend_ports_function: function to extend the ports for a component to ensure it goes beyond the PML
        layer_to_thickness_nm: Dict of layer number (int, int) to thickness (nm)
        res: resolution (pixels/um) For example: (10: 100nm step size)
        t_clad_top: thickness for cladding above core
        t_clad_bot: thickness for cladding below core
        tpml: PML thickness (um)
        clad_material: material for cladding
        is_3d: if True runs in 3D
        wavelengths: iterable of wavelengths to simulate
        dfcen: delta frequency
        sidewall_angle: in degrees
        port_source_name: input port name
        port_field_monitor_name:
        port_margin: margin on each side of the port
        distance_source_to_monitors: in (um) source goes before

    Returns:
        sim: simulation object

    Make sure you visualize the simulation region with gdsfactory before you simulate a component

    .. code::

        import pp
        import gtidy as gm

        c = pp.components.bend_circular()

    """
    assert port_source_name in component.ports
    assert isinstance(
        component, Component
    ), f"component needs to be a gdsfactory Component, got Type {type(component)}"

    component = component.copy()
    component.x = 0
    component.y = 0

    component_extended = (
        pp.extend.extend_ports(component=component, length=extend_ports_length)
        if extend_ports_length
        else component
    )

    pp.show(component_extended)
    component_extended.flatten()

    t_core = max(layer_to_thickness_nm.values()) * 1e-3
    cell_thickness = tpml + t_clad_bot + t_core + t_clad_top + tpml if is_3d else 0

    sim_size = [component.xsize + 2 * tpml, component.ysize + 2 * tpml, cell_thickness]

    geometry = td.GdsSlab(
        material=mat_wg,
        gds_cell=component_extended,
        gds_layer=1,
        z_cent=wg_height / 2,
        z_size=wg_height,
    )

    # Add source
    port = component.ports[port_source_name]
    angle = port.orientation
    width = port.width + 2 * port_margin
    size_x = width * abs(np.sin(angle * np.pi / 180))
    size_y = width * abs(np.cos(angle * np.pi / 180))
    size_x = 0 if size_x < 0.001 else size_x
    size_y = 0 if size_y < 0.001 else size_y
    size_z = cell_thickness - 2 * tpml if is_3d else 20
    size = [size_x, size_y, size_z]
    center = port.center.tolist() + [0]  # (x, y, z=0)

    msource = td.ModeSource(
        size=size,
        center=center,
        source_time=td.GaussianPulse(frequency=freq0, fwidth=fwidth),
        direction="forward",
    )

    # Add port monitors
    monitors = {}
    for port_name in component.ports.keys():
        port = component.ports[port_name]
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
        center=[0, 0, wg_height / 2], size=[sim_size[0], sim_size[1], 0], freqs=[freq0]
    )

    sim = td.Simulation(
        size=sim_size,
        mesh_step=mesh_step,
        structures=[geometry],
        sources=[msource],
        monitors=[domain_monitor] + list(monitors.values()),
        run_time=20 / fwidth,
        pml_layers=[12, 12, 12],
    )
    # set the modes
    sim.compute_modes(msource, Nmodes=2)
    sim.set_mode(msource, mode_ind=0)

    # export simulation to run, get taskID to refer to later
    project = web.new_project(sim.export(), task_name=component.name)
    taskId = project["taskId"]
    return sim, taskId


if __name__ == "__main__":

    c = pp.components.mmi1x2()
    c = pp.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    c = pp.components.straight(length=2)
    c = pp.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    c = pp.components.bend_circular(radius=2)
    # c = pp.add_padding(c, default=0, bottom=2, right=2, layers=[(100, 0)])

    sim, task_id = get_simulation(c, is_3d=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    sim.viz_mat_2D(normal="z", position=wg_height / 2, ax=ax1)
    sim.viz_mat_2D(normal="x", ax=ax2, source_alpha=1)
    ax2.set_xlim([-3, 3])
    plt.show()
