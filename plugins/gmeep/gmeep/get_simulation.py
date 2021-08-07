"""
Returns simulation from component

FIXME, zmin_um does not work

"""
from typing import Any, Dict, Optional, Tuple
import warnings
import pydantic

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.extension import move_polar_rad_copy
from gmeep.materials import get_material

mp.verbosity(0)


LAYER_TO_THICKNESS_NM = {(1, 0): 220.0}
LAYER_TO_MATERIAL = {(1, 0): "Si"}
LAYER_TO_ZMIN = {(1, 0): "Si"}
LAYER_TO_SIDEWALL_ANGLE = {(1, 0): "Si"}


@pydantic.validate_arguments
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
    wl_min: float = 1.5,
    wl_max: float = 1.6,
    wl_steps: int = 50,
    dfcen: float = 0.2,
    port_source_name: str = "W0",
    port_field_monitor_name: str = "E0",
    port_margin: float = 0.5,
    distance_source_to_monitors: float = 0.2,
) -> Dict[str, Any]:
    """Returns Simulation dict from gdsfactory.component

    based on meep directional coupler example
    https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/

    https://support.lumerical.com/hc/en-us/articles/360042095873-Metamaterial-S-parameter-extraction

    Args:
        component: gf.Component
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

    Make sure you visualize the simulation region with gf.before you simulate a component

    .. code::

        import gdsfactory as gf
        import gmeep as gm

        c = gf.components.bend_circular()
        margin = 2
        cm = gm.add_monitors(c)
        gf.show(cm)

    """
    wavelengths = np.linspace(wl_min, wl_max, wl_steps)
    if port_source_name not in component.ports:
        warnings.warn(
            f"port_source_name={port_source_name} not in {component.ports.keys()}"
        )
        port_source = component.get_ports_list()[0]
        port_source_name = port_source.name
        warnings.warn(f"Selecting port_source_name={port_source_name} instead.")

    if port_field_monitor_name not in component.ports:
        warnings.warn(
            f"port_field_monitor_name={port_field_monitor_name} not in {component.ports.keys()}"
        )
        port_field_monitor = (
            component.get_ports_list()[0]
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

    component = component.copy()
    component.x = 0
    component.y = 0

    component_extended = (
        gf.extend.extend_ports(component=component, length=extend_ports_length)
        if extend_ports_length
        else component
    )

    gf.show(component_extended)
    component_extended.flatten()
    # geometry_center = [component_extended.x, component_extended.y]
    # geometry_center = [0, 0]
    # print(geometry_center)

    t_core = max(layer_to_thickness_nm.values()) * 1e-3
    cell_thickness = tpml + t_clad_bot + t_core + t_clad_top + tpml if is_3d else 0

    cell_size = mp.Vector3(
        component.xsize + 2 * tpml,
        component.ysize + 2 * tpml,
        cell_thickness,
    )

    geometry = []
    layer_to_polygons = component_extended.get_polygons(by_spec=True)
    for layer, polygons in layer_to_polygons.items():
        if layer in layer_to_thickness_nm and layer in layer_to_material:
            height = layer_to_thickness_nm[layer] * 1e-3 if is_3d else mp.inf
            zmin_um = layer_to_zmin_nm[layer] * 1e-3 if is_3d else 0
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

    field_monitor_port = component.ports[port_field_monitor_name]
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
    for port_name in component.ports.keys():
        port = component.ports[port_name]
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

    c = gf.components.bend_circular(radius=2)
    c = gf.add_padding(c, default=0, bottom=2, right=2, layers=[(100, 0)])

    c = gf.components.straight(length=2)
    c = gf.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    c = gf.components.mmi1x2()
    c = gf.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    sim_dict = get_simulation(c, is_3d=False)
    sim = sim_dict["sim"]
    sim.plot2D()

    # sim_dict = get_simulation(c, is_3d=True)
    # center = (0, 0, 0)
    # size = (0, 2, 2)
    # sim.plot2D(output_plane=mp.Volume(center=center, size=size))

    plt.show()
