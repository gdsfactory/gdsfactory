"""Returns simulation from component

FIXME, zmin_um does not work
"""
import warnings
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pydantic

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.extension import move_polar_rad_copy
from gdsfactory.simulation.gmeep.get_material import get_material
from gdsfactory.tech import LAYER_STACK, LayerStack

mp.verbosity(0)


LAYER_TO_THICKNESS = {(1, 0): 220e-3}
LAYER_TO_MATERIAL = {(1, 0): "Si"}
LAYER_TO_ZMIN = {(1, 0): "Si"}
LAYER_TO_SIDEWALL_ANGLE = {(1, 0): "Si"}


MATERIAL_NAME_TO_MEEP = {
    "si": "Si",
    "sio2": "SiO2",
    "sin": "Si3N4",
}


@pydantic.validate_arguments
def get_simulation(
    component: Component,
    extend_ports_length: Optional[float] = 4.0,
    layer_stack: LayerStack = LAYER_STACK,
    res: int = 20,
    t_clad_top: float = 1.0,
    t_clad_bot: float = 1.0,
    tpml: float = 1.0,
    clad_material: str = "SiO2",
    is_3d: bool = False,
    wl_min: float = 1.5,
    wl_max: float = 1.6,
    wl_steps: int = 50,
    dfcen: float = 0.2,
    port_source_name: str = 1,
    port_field_monitor_name: str = 2,
    port_margin: float = 0.5,
    distance_source_to_monitors: float = 0.2,
) -> Dict[str, Any]:
    """Returns Simulation dict from gdsfactory.component

    based on meep directional coupler example
    https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/

    https://support.lumerical.com/hc/en-us/articles/360042095873-Metamaterial-S-parameter-extraction

    Args:
        component: gf.Component
        extend_ports_function: to extend ports beyond the PML
        layer_to_thickness: Dict of layer number (int, int) to thickness (um)
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

    Make sure you review the simulation before you simulate a component

    .. code::

        import gdsfactory as gf
        import gdsfactory.simulation.meep as gm

        c = gf.components.bend_circular()
        margin = 2
        cm = gm.add_monitors(c)
        gf.show(cm)

    """
    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_material = layer_stack.get_layer_to_material()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    layer_to_sidewall_angle = layer_stack.get_layer_to_sidewall_angle()

    component_ref = component.ref()
    component_ref.x = 0
    component_ref.y = 0

    wavelengths = np.linspace(wl_min, wl_max, wl_steps)
    if port_source_name not in component_ref.ports:
        warnings.warn(
            f"port_source_name={port_source_name} not in {component.ports.keys()}"
        )
        port_source = component_ref.get_ports_list()[0]
        port_source_name = port_source.name
        warnings.warn(f"Selecting port_source_name={port_source_name} instead.")

    if port_field_monitor_name not in component_ref.ports:
        port_names = list(component_ref.ports.keys())
        warnings.warn(
            f"port_field_monitor_name={port_field_monitor_name} not in {port_names}"
        )
        port_field_monitor = (
            component_ref.get_ports_list()[0]
            if len(component.ports) < 2
            else component.get_ports_list()[1]
        )
        port_field_monitor_name = port_field_monitor.name
        warnings.warn(
            f"Selecting port_field_monitor_name={port_field_monitor_name} instead."
        )

    assert isinstance(
        component, Component
    ), f"component needs to be a gf.Component, got Type {type(component)}"

    component_extended = (
        gf.components.extension.extend_ports(
            component=component, length=extend_ports_length, centered=True
        )
        if extend_ports_length
        else component
    )
    gf.show(component_extended)

    component_extended.flatten()
    component_extended = component_extended.ref()

    # geometry_center = [component_extended.x, component_extended.y]
    # geometry_center = [0, 0]
    # print(geometry_center)

    layers_thickness = [
        layer_to_thickness[layer]
        for layer in component.layers
        if layer in layer_to_thickness
    ]

    t_core = max(layers_thickness)
    cell_thickness = tpml + t_clad_bot + t_core + t_clad_top + tpml if is_3d else 0

    cell_size = mp.Vector3(
        component.xsize + 2 * tpml,
        component.ysize + 2 * tpml,
        cell_thickness,
    )

    geometry = []
    layer_to_polygons = component_extended.get_polygons(by_spec=True)
    for layer, polygons in layer_to_polygons.items():
        if layer in layer_to_thickness and layer in layer_to_material:
            height = layer_to_thickness[layer] if is_3d else mp.inf
            zmin_um = layer_to_zmin[layer] if is_3d else 0
            # center = mp.Vector3(0, 0, (zmin_um + height) / 2)

            for polygon in polygons:
                vertices = [mp.Vector3(p[0], p[1], zmin_um) for p in polygon]
                material_name = layer_to_material[layer]
                material = get_material(name=material_name)
                geometry.append(
                    mp.Prism(
                        vertices=vertices,
                        height=height,
                        sidewall_angle=layer_to_sidewall_angle[layer],
                        material=material,
                        # center=center
                    )
                )

    freqs = 1 / wavelengths
    fcen = np.mean(freqs)
    frequency_width = dfcen * fcen

    # Add source
    port = component_ref.ports[port_source_name]
    angle = port.orientation
    width = port.width + 2 * port_margin
    size_x = width * abs(np.sin(angle * np.pi / 180))
    size_y = width * abs(np.cos(angle * np.pi / 180))
    size_x = 0 if size_x < 0.001 else size_x
    size_y = 0 if size_y < 0.001 else size_y
    size_z = cell_thickness - 2 * tpml if is_3d else 20
    size = [size_x, size_y, size_z]
    center = port.center.tolist() + [0]  # (x, y, z=0)

    field_monitor_port = component_ref.ports[port_field_monitor_name]
    field_monitor_point = field_monitor_port.center.tolist() + [0]  # (x, y, z=0)

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=frequency_width),
            size=size,
            center=center,
            eig_band=1,
            eig_parity=mp.NO_PARITY if is_3d else mp.EVEN_Y + mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    sim = mp.Simulation(
        resolution=res,
        cell_size=cell_size,
        boundary_layers=[mp.PML(tpml)],
        sources=sources,
        geometry=geometry,
        default_material=get_material(name=clad_material),
        # geometry_center=geometry_center,
    )

    # Add port monitors dict
    monitors = {}
    for port_name in component_ref.ports.keys():
        port = component_ref.ports[port_name]
        angle = port.orientation
        width = port.width + 2 * port_margin
        size_x = width * abs(np.sin(angle * np.pi / 180))
        size_y = width * abs(np.cos(angle * np.pi / 180))
        size_x = 0 if size_x < 0.001 else size_x
        size_y = 0 if size_y < 0.001 else size_y
        size = mp.Vector3(size_x, size_y, size_z)
        size = [size_x, size_y, size_z]

        # if monitor has a source move monitor inwards
        length = -distance_source_to_monitors if port_name == port_source_name else 0
        xy_shifted = move_polar_rad_copy(
            np.array(port.center), angle=angle * np.pi / 180, length=length
        )
        center = xy_shifted.tolist() + [0]  # (x, y, z=0)
        m = sim.add_mode_monitor(freqs, mp.ModeRegion(center=center, size=size))
        m.z = 0
        monitors[port_name] = m
    return dict(
        sim=sim,
        cell_size=cell_size,
        freqs=freqs,
        monitors=monitors,
        sources=sources,
        field_monitor_point=field_monitor_point,
        port_source_name=port_source_name,
    )


if __name__ == "__main__":
    c = gf.components.straight(length=2)
    c = gf.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    c = gf.components.mmi1x2()
    c = gf.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    c = gf.components.bend_circular(radius=2)
    c = gf.add_padding(c, default=0, bottom=2, right=2, layers=[(100, 0)])

    sim_dict = get_simulation(c, is_3d=False)
    sim = sim_dict["sim"]

    sim.plot2D()  # plot top view

    # center = (0, 0, 0)
    # size = sim.cell_size
    # sim.plot2D(
    #     output_plane=mp.Volume(center=center, size=(0, size[1], size[2]))
    # )  # plot xsection

    plt.show()
