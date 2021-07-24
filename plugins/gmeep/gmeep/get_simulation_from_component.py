"""Compute and save Sparameters using Meep
"""

import pathlib
from pathlib import Path, PosixPath
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd
import pp
from meep.geom import Medium
from numpy import ndarray
from pp.component import Component
from pp.components.extension import move_polar_rad_copy
from pp.sp.get_sparameters_path import get_sparameters_path

from gmeep.config import PATH

mp.verbosity(0)


MATERIAL_TO_MEDIUM = {
    "si": mp.Medium(epsilon=3.44 ** 2),
    "sin": mp.Medium(epsilon=2.0 ** 2),
    "sio2": mp.Medium(epsilon=1.44 ** 2),
}
LAYER_TO_MATERIAL = {(1, 0): "si"}
LAYER_TO_THICKNESS_NM = {(1, 0): 220.0}


def get_simulation_from_component(
    component: Component,
    extend_ports_function: Callable = pp.extend.extend_ports,
    layer_to_thickness_nm: Dict[Tuple[int, int], float] = LAYER_TO_THICKNESS_NM,
    layer_to_material: Dict[Tuple[int, int], str] = LAYER_TO_MATERIAL,
    res: int = 20,
    t_clad_top: float = 1.0,
    t_clad_bot: float = 1.0,
    tpml: float = 1.0,
    clad_material: Medium = mp.Medium(epsilon=2.25),
    is_3d: bool = False,
    wavelengths: ndarray = np.linspace(1.5, 1.6, 50),
    dfcen: float = 0.2,
    sidewall_angle: float = 0.0,
    port_source_name: str = "W0",
    port_field_monitor_name: str = "E0",
    port_margin: float = 0.5,
    distance_source_to_monitors: float = 0.2,
) -> Dict[str, Any]:
    """Returns Simulation dict from gdsfactory component

    based on meep directional coupler example
    https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/

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
        import gmeep as gm

        c = pp.components.bend_circular()
        margin = 2
        cm = gm.add_monitors(c)
        pp.show(cm)

    """
    assert port_source_name in component.ports
    assert isinstance(
        component, Component
    ), f"component needs to be a gdsfactory Component, got Type {type(component)}"

    component.x = 0
    component.y = 0

    # extend waveguides beyond PML
    component_extended = extend_ports_function(component=component, length=tpml)

    pp.show(component_extended)
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
            t_core = (
                layer_to_thickness_nm[layer] * 1e-3 if is_3d else mp.inf
            )  # nm to um

            for polygon in polygons:
                vertices = [mp.Vector3(p[0], p[1]) for p in polygon]
                material_str = layer_to_material[layer]
                material = MATERIAL_TO_MEDIUM[material_str]
                geometry.append(
                    mp.Prism(
                        vertices=vertices,
                        height=t_core,
                        sidewall_angle=sidewall_angle,
                        material=material,
                    )
                )

    freqs = 1 / wavelengths
    fcen = np.mean(freqs)
    frequency_width = dfcen * fcen

    # Add source
    # define sources and monitors size
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
        default_material=clad_material,
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
        field_monitor_point=field_monitor_point,
    )


def write_sparameters(
    component: Component,
    dirpath: PosixPath = PATH.sparameters,
    layer_to_thickness_nm: Dict[Tuple[int, int], float] = {(1, 0): 220.0},
    layer_to_material: Dict[Tuple[int, int], str] = LAYER_TO_MATERIAL,
    filepath: Optional[Path] = None,
    overwrite: bool = False,
    **settings,
) -> pd.DataFrame:
    """Compute Sparameters and writes them in CSV filepath.

    Args:
        component: to simulate.
        dirpath: directory to store Sparameters
        layer_to_thickness_nm: GDS layer (int, int) to thickness
        layer_to_material: GDS layer (int, int) to material string ('si', 'sio2', ...)
        filepath: to store pandas Dataframe
        overwrite: overwrites
        **settings: sim settings

    Returns:
        S parameters pandas Dataframe

    """
    filepath = filepath or get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_to_material,
        layer_to_thickness_nm=layer_to_thickness_nm,
    )
    filepath = pathlib.Path(filepath)
    if filepath.exists() and not overwrite:
        return pd.read_csv(filepath)

    sim_dict = get_simulation_from_component(
        component=component,
        layer_to_thickness_nm=layer_to_thickness_nm,
        layer_to_material=layer_to_material,
        **settings,
    )

    sim = sim_dict["sim"]
    monitors = sim_dict["monitors"]
    freqs = sim_dict["freqs"]
    field_monitor_point = sim_dict["field_monitor_point"]
    wavelengths = 1 / freqs
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            dt=50, c=mp.Ez, pt=field_monitor_point, decay_by=1e-9
        )
    )
    # call this function every 50 time spes
    # look at simulation and measure component that we want to measure (Ez component)
    # when field_monitor_point decays below a certain 1e-9 field threshold

    # Calculate the mode overlaps
    nports = len(monitors)
    print((len(freqs), nports, nports))
    S = np.zeros((len(freqs), nports, nports))
    a = {}
    b = {}

    for port_name, monitor in monitors.items():
        m_results = sim.get_eigenmode_coefficients(monitor, [1]).alpha

        # Parse out the overlaps
        a[port_name] = m_results[:, :, 0]  # forward wave
        b[port_name] = m_results[:, :, 1]  # backward wave

    for i, port_name_i in enumerate(monitors.keys()):
        for j, port_name_j in enumerate(monitors.keys()):
            S[:, i, j] = np.squeeze(a[port_name_j] / b[port_name_i])
            S[:, j, i] = np.squeeze(a[port_name_i] / b[port_name_j])

    # for port_name in monitor.keys():
    #     a1 = m1_results[:, :, 0]  # forward wave
    #     b1 = m1_results[:, :, 1]  # backward wave
    #     a2 = m2_results[:, :, 0]  # forward wave
    #     # b2 = m2_results[:, :, 1]  # backward wave

    #     # Calculate the actual scattering parameters from the overlaps
    #     s11 = np.squeeze(b1 / a1)
    #     s12 = np.squeeze(a2 / a1)

    r = dict(wavelengths=wavelengths)
    keys = [key for key in r.keys() if key.startswith("s")]
    s = {f"{key}a": list(np.unwrap(np.angle(r[key].flatten()))) for key in keys}
    s.update({f"{key}m": list(np.abs(r[key].flatten())) for key in keys})
    s.update(wavelengths=wavelengths)
    s.update(freqs=freqs)
    df = pd.DataFrame(s)
    # df = df.set_index(df.wavelength)
    df.to_csv(filepath, index=False)

    return df


def plot_sparameters(df: pd.DataFrame) -> None:
    """Plot Sparameters from a Pandas DataFrame."""
    wavelengths = df["wavelengths"]
    for key in df.keys():
        if key.endswith("m"):
            plt.plot(
                wavelengths,
                df[key],
                "-o",
                label="key",
            )
    plt.ylabel("Power (dB)")
    plt.xlabel(r"Wavelength ($\mu$m)")
    plt.legend()
    plt.grid(True)


def write_sparameters_sweep(
    component,
    dirpath: PosixPath = PATH.sparameters,
    layer_to_thickness_nm: Dict[Tuple[int, int], float] = {(1, 0): 220.0},
    layer_to_material: Dict[Tuple[int, int], str] = LAYER_TO_MATERIAL,
    **kwargs,
):
    """From gdsfactory component writes Sparameters for all the ports."""
    filepath_lumerical = get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_to_material,
        layer_to_thickness_nm=layer_to_thickness_nm,
    )
    for port_source_name in component.ports.keys():
        sim_dict = get_simulation_from_component(
            port_source_name=port_source_name,
            layer_to_thickness_nm=layer_to_thickness_nm,
            layer_to_material=layer_to_material,
            **kwargs,
        )
        filepath = filepath_lumerical.with_suffix(f"{port_source_name}.csv")
        write_sparameters(sim_dict, filepath=filepath)


if __name__ == "__main__":

    c = pp.c.bend_circular(radius=2)
    c = pp.add_padding(c, default=0, bottom=2, right=2, layers=[(100, 0)])

    c = pp.c.mmi1x2()
    c = pp.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    c = pp.c.straight(length=2)
    c = pp.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    sim_dict = get_simulation_from_component(c, is_3d=False)
    df = write_sparameters(c)
    plot_sparameters(df)
    plt.show()
