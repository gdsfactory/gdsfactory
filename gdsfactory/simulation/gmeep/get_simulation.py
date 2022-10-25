"""Returns simulation from component."""
import inspect
import warnings
from typing import Any, Dict, Optional, Union

import meep as mp
import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.extension import move_polar_rad_copy
from gdsfactory.pdk import get_layer_stack
from gdsfactory.simulation.gmeep.get_material import get_material
from gdsfactory.simulation.gmeep.get_meep_geometry import (
    get_meep_geometry_from_component,
)
from gdsfactory.tech import LayerStack

mp.verbosity(0)

sig = inspect.signature(mp.Simulation)
settings_meep = set(sig.parameters.keys())


def get_simulation(
    component: Component,
    resolution: int = 30,
    extend_ports_length: Optional[float] = 10.0,
    layer_stack: Optional[LayerStack] = None,
    zmargin_top: float = 3.0,
    zmargin_bot: float = 3.0,
    tpml: float = 1.5,
    clad_material: str = "SiO2",
    is_3d: bool = False,
    wavelength_start: float = 1.5,
    wavelength_stop: float = 1.6,
    wavelength_points: int = 50,
    dfcen: float = 0.2,
    port_source_name: str = "o1",
    port_margin: float = 3,
    distance_source_to_monitors: float = 0.2,
    port_source_offset: float = 0,
    port_monitor_offset: float = 0,
    dispersive: bool = False,
    material_name_to_meep: Optional[Dict[str, Union[str, float]]] = None,
    **settings,
) -> Dict[str, Any]:
    r"""Returns Simulation dict from gdsfactory Component.

    based on meep directional coupler example
    https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/

    https://support.lumerical.com/hc/en-us/articles/360042095873-Metamaterial-S-parameter-extraction

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
             |ymargin              |         |
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
        component: gdsfactory Component.
        resolution: in pixels/um (20: for coarse, 120: for fine).
        extend_ports_length: to extend ports beyond the PML.
        layer_stack: contains layer to thickness, zmin and material.
            Defaults to active pdk.layer_stack.
        zmargin_top: thickness for cladding above core.
        zmargin_bot: thickness for cladding below core.
        tpml: PML thickness (um).
        clad_material: material for cladding.
        is_3d: if True runs in 3D.
        wavelength_start: wavelength min (um).
        wavelength_stop: wavelength max (um).
        wavelength_points: wavelength steps.
        dfcen: delta frequency.
        port_source_name: input port name.
        port_margin: margin on each side of the port.
        distance_source_to_monitors: in (um) source goes before.
        port_source_offset: offset between source GDS port and source MEEP port.
        port_monitor_offset: offset between monitor GDS port and monitor MEEP port.
        dispersive: use dispersive material models (requires higher resolution).
        material_name_to_meep: map layer_stack names with meep material database name
            or refractive index. dispersive materials have a wavelength dependent index.

    Keyword Args:
        settings: extra simulation settings (resolution, symmetries, etc.)

    Returns:
        simulation dict: sim, monitors, sources.

    Make sure you review the simulation before you simulate a component

    .. code::

        import gdsfactory as gf
        import gdsfactory.simulation.meep as gm

        c = gf.components.bend_circular()
        gm.write_sparameters_meep(c, run=False)

    """
    for setting in settings:
        if setting not in settings_meep:
            raise ValueError(f"{setting!r} not in {settings_meep}")

    layer_stack = layer_stack or get_layer_stack()
    layer_to_thickness = layer_stack.get_layer_to_thickness()

    component_ref = component.ref()
    component_ref.x = 0
    component_ref.y = 0

    wavelength = (wavelength_start + wavelength_stop) / 2

    wavelengths = np.linspace(wavelength_start, wavelength_stop, wavelength_points)
    port_names = list(component_ref.ports.keys())

    if port_source_name not in port_names:
        warnings.warn(f"port_source_name={port_source_name!r} not in {port_names}")
        port_source = component_ref.get_ports_list()[0]
        port_source_name = port_source.name
        warnings.warn(f"Selecting port_source_name={port_source_name!r} instead.")

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

    # geometry_center = [component_extended.x, component_extended.y]
    # geometry_center = [0, 0]
    # print(geometry_center)

    layers_thickness = [
        layer_to_thickness[layer]
        for layer in component.layers
        if layer in layer_to_thickness
    ]

    if layers_thickness is None:
        raise ValueError(
            f"Component layers {component.layers} not in {layer_to_thickness.keys()}. "
            "Did you passed the correct layer_stack?"
        )

    t_core = max(layers_thickness)
    cell_thickness = tpml + zmargin_bot + t_core + zmargin_top + tpml if is_3d else 0

    cell_size = mp.Vector3(
        component.xsize + 2 * tpml,
        component.ysize + 2 * tpml,
        cell_thickness,
    )

    geometry = get_meep_geometry_from_component(
        component=component_extended,
        layer_stack=layer_stack,
        material_name_to_meep=material_name_to_meep,
        wavelength=wavelength,
        is_3d=is_3d,
        dispersive=dispersive,
    )

    freqs = 1 / wavelengths
    fcen = np.mean(freqs)
    frequency_width = dfcen * fcen

    # Add source
    port = component_ref.ports[port_source_name]
    angle_rad = np.radians(port.orientation)
    width = port.width + 2 * port_margin
    size_x = width * abs(np.sin(angle_rad))
    size_y = width * abs(np.cos(angle_rad))
    size_x = 0 if size_x < 0.001 else size_x
    size_y = 0 if size_y < 0.001 else size_y
    size_z = cell_thickness - 2 * tpml if is_3d else 20
    size = [size_x, size_y, size_z]
    xy_shifted = move_polar_rad_copy(
        np.array(port.center), angle=angle_rad, length=port_source_offset
    )
    center = xy_shifted.tolist() + [0]  # (x, y, z=0)

    if np.isclose(port.orientation, 0):
        direction = mp.X
    elif np.isclose(port.orientation, 90):
        direction = mp.Y
    elif np.isclose(port.orientation, 180):
        direction = mp.X
    elif np.isclose(port.orientation, 270):
        direction = mp.Y
    else:
        raise ValueError(
            f"Port source {port_source_name!r} orientation {port.orientation} "
            "not 0, 90, 180, 270 degrees"
        )

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=frequency_width),
            size=size,
            center=center,
            eig_band=1,
            eig_parity=mp.NO_PARITY if is_3d else mp.EVEN_Y + mp.ODD_Z,
            eig_match_freq=True,
            eig_kpoint=-1 * mp.Vector3(x=1).rotate(mp.Vector3(z=1), angle_rad),
            direction=direction,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(tpml)],
        sources=sources,
        geometry=geometry,
        default_material=get_material(
            name=clad_material,
            material_name_to_meep=material_name_to_meep,
            wavelength=wavelength,
        ),
        resolution=resolution,
        **settings,
    )

    # Add port monitors dict
    monitors = {}
    for port_name in component_ref.ports.keys():
        port = component_ref.ports[port_name]
        angle_rad = np.radians(port.orientation)
        width = port.width + 2 * port_margin
        size_x = width * abs(np.sin(angle_rad))
        size_y = width * abs(np.cos(angle_rad))
        size_x = 0 if size_x < 0.001 else size_x
        size_y = 0 if size_y < 0.001 else size_y
        size = mp.Vector3(size_x, size_y, size_z)
        size = [size_x, size_y, size_z]

        # if monitor has a source move monitor inwards
        length = (
            -distance_source_to_monitors + port_source_offset
            if port_name == port_source_name
            else port_monitor_offset
        )
        xy_shifted = move_polar_rad_copy(
            np.array(port.center), angle=angle_rad, length=length
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
        port_source_name=port_source_name,
        initialized=False,
    )


sig = inspect.signature(get_simulation)
settings_get_simulation = set(sig.parameters.keys()).union(settings_meep)


if __name__ == "__main__":
    c = gf.components.straight(length=2, width=0.5)
    sim_dict = get_simulation(
        c,
        is_3d=False,
        # resolution=50,
        # port_source_offset=-0.1,
        # port_field_monitor_offset=-0.1,
        # port_margin=2.5,
    )
    # sim.plot3D()
    # sim.plot2D()  # plot top view (is_3D needs to be False)
    # Plot monitor cross-section (is_3D needs to be True)

    # sim.init_sim()
    # eps_data = sim.get_epsilon()

    # from mayavi import mlab
    # s = mlab.contour3d(eps_data, colormap="YlGnBu")
    # mlab.show()

    print(settings_get_simulation)
