from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd
import pydantic

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.simulation.gmeep.add_monitors import add_monitors
from gdsfactory.simulation.gmeep.get_material import get_material

mp.verbosity(0)


@pydantic.validate_arguments
def get_transmission_2ports(
    component: Component,
    extend_ports_length: Optional[float] = 4.0,
    layer_core: int = 1,
    layer_source: int = 110,
    layer_monitor1: int = 101,
    layer_monitor2: int = 102,
    layer_simulation_region: int = 2,
    resolution: int = 20,
    t_clad_bot: float = 1.0,
    t_core: float = 0.22,
    t_clad_top: float = 1.0,
    dpml: int = 1,
    clad_material: str = "SiO2",
    core_material: str = "Si",
    is_3d: bool = False,
    run: bool = True,
    wavelength_min: float = 1.5,
    wavelength_max: float = 1.6,
    wavelength_points: int = 50,
    field_monitor_point: Tuple[int, int, int] = (0, 0, 0),
    dfcen: float = 0.2,
) -> Dict[str, Any]:
    """Returns dict with Sparameters for a 2port gf.component

    requires source and  port monitors in the GDS

    based on meep directional coupler example
    https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/

    https://support.lumerical.com/hc/en-us/articles/360042095873-Metamaterial-S-parameter-extraction

    Args:
        component: gf.Component
        extend_ports_function: function to extend ports beyond the PML
        layer_core: GDS layer for the Component material
        layer_source: for the source monitor
        layer_monitor1: monitor layer for port 1
        layer_monitor2: monitor layer for port 2
        layer_simulation_region: for simulation region
        resolution: resolution (pixels/um) For example: (10: 100nm step size)
        t_clad_bot: thickness for cladding below core
        t_core: thickness of the core material
        t_clad_top: thickness for cladding above core
        dpml: PML thickness (um)
        clad_material: material for cladding
        core_material: material for core
        is_3d: if True runs in 3D
        run: if True runs simulation, False only build simulation
        wavelengths: iterable of wavelengths to simulate
        field_monitor_point: monitors field and stops simulation after field decays by 1e-9
        dfcen: delta frequency

    Returns:
        Dict:
            sim: simulation object

    Make sure you visualize the simulation with before you run

    .. code::

        import gdsfactory as gf
        import gdsfactory.simulation.meep as gm

        component = gf.components.bend_circular()
        margin = 2
        cm = gm.add_monitors(component)
        cm.show()

    """
    clad_material = get_material(name=clad_material)
    core_material = get_material(name=core_material)

    assert isinstance(
        component, Component
    ), f"component needs to be a Component, got Type {type(component)}"
    if extend_ports_length:
        component = gf.components.extension.extend_ports(
            component=component, length=extend_ports_length, centered=True
        )
    component.flatten()
    gdspath = component.write_gds()
    gdspath = str(gdspath)

    wavelengths = np.linspace(wavelength_min, wavelength_max, wavelength_points)
    freqs = 1 / wavelengths
    fcen = np.mean(freqs)
    frequency_width = dfcen * fcen
    cell_thickness = dpml + t_clad_bot + t_core + t_clad_top + dpml

    cell_zmax = 0.5 * cell_thickness if is_3d else 0
    cell_zmin = -0.5 * cell_thickness if is_3d else 0

    core_zmax = 0.5 * t_core if is_3d else 10
    core_zmin = -0.5 * t_core if is_3d else -10

    geometry = mp.get_GDSII_prisms(
        core_material, gdspath, layer_core, core_zmin, core_zmax
    )
    cell = mp.GDSII_vol(gdspath, layer_core, cell_zmin, cell_zmax)
    sim_region = mp.GDSII_vol(gdspath, layer_simulation_region, cell_zmin, cell_zmax)

    cell.size = mp.Vector3(
        sim_region.size[0] + 2 * dpml, sim_region.size[1] + 2 * dpml, sim_region.size[2]
    )
    cell_size = cell.size

    zsim = t_core + t_clad_top + t_clad_bot + 2 * dpml
    m_zmin = -zsim / 2
    m_zmax = +zsim / 2
    src_vol = mp.GDSII_vol(gdspath, layer_source, m_zmin, m_zmax)

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=frequency_width),
            size=src_vol.size,
            center=src_vol.center,
            eig_band=1,
            eig_parity=mp.NO_PARITY if is_3d else mp.EVEN_Y + mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=[mp.PML(dpml)],
        sources=sources,
        geometry=geometry,
        default_material=clad_material,
    )
    sim_settings = dict(
        resolution=resolution,
        cell_size=cell_size,
        fcen=fcen,
        field_monitor_point=field_monitor_point,
        layer_core=layer_core,
        t_clad_bot=t_clad_bot,
        t_core=t_core,
        t_clad_top=t_clad_top,
        is_3d=is_3d,
        dmp=dpml,
    )

    m1_vol = mp.GDSII_vol(gdspath, layer_monitor1, m_zmin, m_zmax)
    m2_vol = mp.GDSII_vol(gdspath, layer_monitor2, m_zmin, m_zmax)
    m1 = sim.add_mode_monitor(
        freqs,
        mp.ModeRegion(center=m1_vol.center, size=m1_vol.size),
    )
    m1.z = 0
    m2 = sim.add_mode_monitor(
        freqs,
        mp.ModeRegion(center=m2_vol.center, size=m2_vol.size),
    )
    m2.z = 0

    # if 0:
    #     ''' Useful for debugging.  '''
    #     sim.run(until=50)
    #     sim.plot2D(fields=mp.Ez)
    #     plt.show()
    #     quit()

    r = dict(sim=sim, cell_size=cell_size, sim_settings=sim_settings)

    if run:
        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                dt=50, c=mp.Ez, pt=field_monitor_point, decay_by=1e-9
            )
        )

        # call this function every 50 time spes
        # look at simulation and measure component that we want to measure (Ez component)
        # when field_monitor_point decays below a certain 1e-9 field threshold

        # Calculate the mode overlaps
        m1_results = sim.get_eigenmode_coefficients(m1, [1]).alpha
        m2_results = sim.get_eigenmode_coefficients(m2, [1]).alpha

        # Parse out the overlaps
        a1 = m1_results[:, :, 0]  # forward wave
        b1 = m1_results[:, :, 1]  # backward wave
        a2 = m2_results[:, :, 0]  # forward wave
        # b2 = m2_results[:, :, 1]  # backward wave

        # Calculate the actual scattering parameters from the overlaps
        s11 = np.squeeze(b1 / a1)
        s12 = np.squeeze(a2 / a1)
        s22 = s11.copy()
        s21 = s12.copy()

        # s22 and s21 requires another simulation, with the source on the other port
        # Luckily, if the device is symmetric, we can assume that s22=s11 and s21=s12.

        # visualize results
        plt.figure()
        plt.plot(
            wavelengths,
            10 * np.log10(np.abs(s11) ** 2),
            "-o",
            label="Reflection",
        )
        plt.plot(
            wavelengths,
            10 * np.log10(np.abs(s12) ** 2),
            "-o",
            label="Transmission",
        )
        plt.ylabel("Power (dB)")
        plt.xlabel(r"Wavelength ($\mu$m)")
        plt.legend()
        plt.grid(True)

        r.update(dict(s11=s11, s12=s12, s21=s21, s22=s22, wavelengths=wavelengths))
        keys = [key for key in r.keys() if key.startswith("S")]
        s = {f"{key}a": list(np.unwrap(np.angle(r[key].flatten()))) for key in keys}
        s_mod = {f"{key}m": list(np.abs(r[key].flatten())) for key in keys}
        s.update(**s_mod)
        s = pd.DataFrame(s)
    return r


def plot2D(results_dict, z=0):
    """Plot a 2D cut of your simulation."""
    sim = results_dict["sim"]
    cell_size = results_dict["cell_size"]
    cell_size.z = 0
    sim.plot2D(
        output_plane=mp.Volume(center=mp.Vector3(), size=cell_size),
        fields=mp.Ez,
        field_parameters={"interpolation": "spline36", "cmap": "RdBu"},
    )


def plot3D(results_dict):
    """Plots 3D simulation in Mayavi."""
    sim = results_dict["sim"]
    sim.plot3D()


def test_waveguide_2D() -> None:
    """Ensure >99% transmission (S21) at 1550nm."""
    c = gf.components.straight(length=2)
    cm = add_monitors(component=c)
    # gf.show(cm)

    r = get_transmission_2ports(cm, is_3d=False, run=True)
    assert 0.99 < np.mean(abs(r["s21"])) < 1.01
    assert 0 < np.mean(abs(r["s11"])) < 0.2


# def test_waveguide_3D() -> None:
#     """Ensure >99% transmission (S21) at 1550nm."""
#     c = gf.components.straight(length=2)
#     cm = add_monitors(component=c)
#     gf.show(cm)

#     r = get_transmission_2ports(cm, is_3d=True, run=True, res=10)
#     assert 0.99 < np.mean(abs(r["s21"])) < 1.01
#     assert 0 < np.mean(abs(r["s11"])) < 0.2


def test_bend_2D():
    """Ensure >99% transmission (S21) at 1550nm."""
    c = gf.components.bend_circular(radius=5)
    cm = add_monitors(component=c)
    # gf.show(cm)

    r = get_transmission_2ports(cm, is_3d=False, run=True)
    assert 0.97 < np.mean(abs(r["s21"])) < 1.01
    assert 0 < np.mean(abs(r["s11"])) < 0.2


if __name__ == "__main__":
    c = gf.components.straight(length=2)
    cm = add_monitors(component=c)
    gf.show(cm)

    r = get_transmission_2ports(cm, run=True)
    print(r)

    # sim = r["sim"]
    # plt.show()
