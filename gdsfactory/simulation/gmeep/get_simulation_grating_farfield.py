"""SMF specs from photonics.byu.edu/FiberOpticConnectors.parts/images/smf28.pdf

MFD:

- 10.4 for Cband
- 9.2 for Oband

TODO:

- verify with lumerical sims
- get Sparameters
- enable mpi run from python

"""
from typing import Any, Dict, Optional

import meep as mp
import numpy as np

from gdsfactory.types import Floats

nm = 1e-3
nSi = 3.48
nSiO2 = 1.44


def fiber_ncore(fiber_numerical_aperture, fiber_nclad):
    return (fiber_numerical_aperture ** 2 + fiber_nclad ** 2) ** 0.5


def get_simulation_grating_farfield(
    period: float = 0.66,
    fill_factor: float = 0.5,
    widths: Optional[Floats] = None,
    gaps: Optional[Floats] = None,
    fiber_angle_deg: float = 20.0,
    fiber_xposition: float = 1.0,
    fiber_core_diameter: float = 10.4,
    fiber_numerical_aperture: float = 0.14,
    fiber_nclad: float = nSiO2,
    res: int = 64,  # pixels/um
    ncore: float = nSi,
    nclad: float = nSiO2,
    nsubstrate: float = nSi,
    n_periods: int = 30,
    box_thickness: float = 2.0,
    clad_thickness: float = 2.0,
    core_thickness: float = 220 * nm,
    etch_depth: float = 70 * nm,
    wavelength_min: float = 1.5,
    wavelength_max: float = 1.6,
    wavelength_points: int = 50,
    dtaper: float = 1,
    # **settings,
) -> Dict[str, Any]:
    """Returns simulation results from grating coupler with fiber.
    na**2 = ncore**2 - nclad**2
    ncore = sqrt(na**2 + ncore**2)

    Args:
        period: grating coupler period
        fill_factor:
        widths: overrides n_periods period and fill_factor
        gaps: overrides n_periods period and fill_factor
        fiber_angle_deg: angle fiber in degrees
        decay_by: 1e-9

    """
    wavelengths = np.linspace(wavelength_min, wavelength_max, wavelength_points)
    wavelength = np.mean(wavelengths)
    freqs = 1 / wavelengths
    widths = widths or n_periods * [period * fill_factor]
    gaps = gaps or n_periods * [period * (1 - fill_factor)]

    settings = dict(
        period=period,
        fill_factor=fill_factor,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xposition=fiber_xposition,
        fiber_core_diameter=fiber_core_diameter,
        fiber_numerical_aperture=fiber_core_diameter,
        fiber_nclad=fiber_nclad,
        res=res,
        ncore=ncore,
        nclad=nclad,
        nsubstrate=nsubstrate,
        n_periods=n_periods,
        box_thickness=box_thickness,
        clad_thickness=clad_thickness,
        etch_depth=etch_depth,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_points=wavelength_points,
        dtaper=dtaper,
        widths=widths,
        gaps=gaps,
    )
    # settings_string = clean_value_name(settings)
    # settings_hash = hashlib.md5(settings_string.encode()).hexdigest()[:8]

    length_grating = np.sum(widths) + np.sum(gaps)

    substrate_thickness = 1.0
    hair = 4
    core_material = mp.Medium(index=ncore)
    clad_material = mp.Medium(index=nclad)
    fiber_angle = np.radians(fiber_angle_deg)

    y_offset = 0

    # Minimally-parametrized computational cell
    # Could be further optimized
    dpml = 1

    # X-domain
    dbufferx = 0.5
    if length_grating + dtaper < 3 * fiber_core_diameter:
        sxy = 3 * fiber_core_diameter + 2 * dbufferx + 2 * dpml
    else:  # Fiber probably to the left
        sxy = 3 / 2 * fiber_core_diameter + length_grating / 2 + 2 * dbufferx + 2 * dpml

    # Useful reference points
    cell_edge_left = -sxy / 2 + dbufferx + dpml
    grating_start = -fiber_xposition

    # Y-domain (using z notation from 3D legacy code)
    dbuffery = 0.5
    sz = (
        2 * dbuffery
        + box_thickness
        + core_thickness
        + hair
        + substrate_thickness
        + 2 * dpml
    )

    # Initialize domain x-z plane simulation
    cell_size = mp.Vector3(sxy, sz)

    # Ports (position, sizes, directions)
    fiber_offset_from_angle = (clad_thickness + core_thickness) * np.tan(fiber_angle)
    fiber_port_center = mp.Vector3(
        (0.5 * sz - dpml + y_offset - 1) * np.sin(fiber_angle)
        + cell_edge_left
        + 3 / 2 * fiber_core_diameter
        - fiber_offset_from_angle,
        0.5 * sz - dpml + y_offset - 1,
    )
    fiber_port_size = mp.Vector3(3 * fiber_core_diameter, 0, 0)
    # fiber_port_direction = mp.Vector3(y=-1).rotate(mp.Vector3(z=1), -1 * fiber_angle)

    waveguide_port_center = mp.Vector3(-sxy / 4)  # grating_start - dtaper, 0)
    waveguide_port_size = mp.Vector3(0, 2 * clad_thickness - 0.2)
    waveguide_port_direction = mp.X

    # Geometry
    fiber_clad = 120
    hfiber_geom = 100  # Some large number to make fiber extend into PML

    fiber_ncore = (fiber_numerical_aperture ** 2 + fiber_nclad ** 2) ** 0.5
    fiber_clad_material = mp.Medium(index=fiber_nclad)
    fiber_core_material = mp.Medium(index=fiber_ncore)

    geometry = []
    # Fiber (defined first to be overridden)
    geometry.append(
        mp.Block(
            material=fiber_clad_material,
            center=mp.Vector3(
                x=grating_start + fiber_xposition - fiber_offset_from_angle
            ),
            size=mp.Vector3(fiber_clad, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    )
    geometry.append(
        mp.Block(
            material=fiber_core_material,
            center=mp.Vector3(
                x=grating_start + fiber_xposition - fiber_offset_from_angle
            ),
            size=mp.Vector3(fiber_core_diameter, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    )

    # clad
    geometry.append(
        mp.Block(
            material=clad_material,
            center=mp.Vector3(0, clad_thickness / 2),
            size=mp.Vector3(mp.inf, clad_thickness),
        )
    )
    # BOX
    geometry.append(
        mp.Block(
            material=clad_material,
            center=mp.Vector3(0, -0.5 * box_thickness),
            size=mp.Vector3(mp.inf, box_thickness),
        )
    )

    # waveguide
    geometry.append(
        mp.Block(
            material=core_material,
            center=mp.Vector3(0, core_thickness / 2),
            size=mp.Vector3(mp.inf, core_thickness),
        )
    )

    # grating etch
    x = grating_start
    for width, gap in zip(widths, gaps):
        geometry.append(
            mp.Block(
                material=clad_material,
                center=mp.Vector3(x + gap / 2, core_thickness - etch_depth / 2),
                size=mp.Vector3(gap, etch_depth),
            )
        )
        x += width + gap

    # Substrate
    geometry.append(
        mp.Block(
            material=mp.Medium(index=nsubstrate),
            center=mp.Vector3(
                0,
                -0.5 * (core_thickness + substrate_thickness + dpml + dbuffery)
                - box_thickness,
            ),
            size=mp.Vector3(mp.inf, substrate_thickness + dpml + dbuffery),
        )
    )

    # PMLs
    boundary_layers = [mp.PML(dpml)]

    # mode frequency
    fcen = 1 / wavelength

    # Waveguide source
    sources_directions = [mp.X]
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
            size=waveguide_port_size,
            center=waveguide_port_center,
            eig_band=1,
            direction=sources_directions[0],
            eig_match_freq=True,
            eig_parity=mp.ODD_Z,
        )
    ]

    # Ports
    waveguide_monitor_port = mp.ModeRegion(
        center=waveguide_port_center + mp.Vector3(x=0.2), size=waveguide_port_size
    )
    fiber_monitor_port = mp.ModeRegion(
        center=fiber_port_center - mp.Vector3(y=0.2), size=fiber_port_size
    )

    sim = mp.Simulation(
        resolution=res,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        geometry=geometry,
        sources=sources,
        dimensions=2,
        eps_averaging=True,
    )
    waveguide_monitor = sim.add_mode_monitor(
        freqs, waveguide_monitor_port, yee_grid=True
    )
    fiber_monitor = sim.add_mode_monitor(freqs, fiber_monitor_port)
    field_monitor_point = (0, 0, 0)

    return dict(
        sim=sim,
        cell_size=cell_size,
        freqs=freqs,
        fcen=fcen,
        waveguide_monitor=waveguide_monitor,
        waveguide_port_direction=waveguide_port_direction,
        fiber_monitor=fiber_monitor,
        fiber_angle_deg=fiber_angle_deg,
        sources=sources,
        field_monitor_point=field_monitor_point,
        initialized=False,
        settings=settings,
    )


def get_port_1D_eigenmode(
    sim_dict,
    band_num=1,
    fiber_angle_deg=15,
):
    """

    Args:
        sim_dict: simulation dict
        band_num: band number to solve for

    Returns:
        Mode object compatible with /modes plugin
    """
    # Initialize
    sim = sim_dict["sim"]
    source = sim_dict["sources"][0]
    waveguide_monitor = sim_dict["waveguide_monitor"]
    fiber_monitor = sim_dict["fiber_monitor"]

    # Obtain source frequency
    fsrc = source.src.frequency

    # Obtain xsection
    center_fiber = fiber_monitor.regions[0].center
    size_fiber = fiber_monitor.regions[0].size
    center_waveguide = waveguide_monitor.regions[0].center
    size_waveguide = waveguide_monitor.regions[0].size

    # Solve for the modes
    if sim_dict["initialized"] is False:
        sim.init_sim()
        sim_dict["initialized"] = True

    # Waveguide
    eigenmode_waveguide = sim.get_eigenmode(
        direction=mp.X,
        where=mp.Volume(center=center_waveguide, size=size_waveguide),
        band_num=band_num,
        kpoint=mp.Vector3(
            fsrc * 3.48, 0, 0
        ),  # Hardcoded index for now, pull from simulation eventually
        frequency=fsrc,
    )
    ys_waveguide = np.linspace(
        center_waveguide.y - size_waveguide.y / 2,
        center_waveguide.y + size_waveguide.y / 2,
        int(sim.resolution * size_waveguide.y),
    )
    x_waveguide = center_waveguide.x

    # Fiber
    eigenmode_fiber = sim.get_eigenmode(
        direction=mp.NO_DIRECTION,
        where=mp.Volume(center=center_fiber, size=size_fiber),
        band_num=band_num,
        kpoint=mp.Vector3(0, fsrc * 1.45, 0).rotate(
            mp.Vector3(z=1), -1 * np.radians(fiber_angle_deg)
        ),  # Hardcoded index for now, pull from simulation eventually
        frequency=fsrc,
    )
    xs_fiber = np.linspace(
        center_fiber.x - size_fiber.x / 2,
        center_fiber.x + size_fiber.x / 2,
        int(sim.resolution * size_fiber.x),
    )
    y_fiber = center_fiber.y

    return (
        x_waveguide,
        ys_waveguide,
        eigenmode_waveguide,
        xs_fiber,
        y_fiber,
        eigenmode_fiber,
    )


def plot(sim):
    """
    sim: simulation object
    """
    sim.plot2D(eps_parameters={"contour": True})
    # plt.colorbar()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # sim_dict = get_GC_simulation(fiber_xposition=5, fiber_angle_deg=15)
    # # plot(sim_dict["sim"])
    # # plt.show()
    # plt.figure()
    # results = {}
    # for angle in [10, 15]:  # np.linspace(0,360,72):
    #     print(angle)
    #     (
    #         x_waveguide,
    #         ys_waveguide,
    #         eigenmode_waveguide,
    #         xs_fiber,
    #         y_fiber,
    #         eigenmode_fiber,
    #     ) = get_port_1D_eigenmode(sim_dict, band_num=1, fiber_angle_deg=angle)
    #     Ez_fiber = np.zeros(len(xs_fiber), dtype=np.complex128)
    #     for i in range(len(xs_fiber)):
    #         Ez_fiber[i] = eigenmode_fiber.amplitude(
    #             mp.Vector3(xs_fiber[i], y_fiber, 0), mp.Ez
    #         )
    #     plt.plot(xs_fiber, np.abs(Ez_fiber))
    # # Ez_waveguide = np.zeros(len(ys_waveguide), dtype=np.complex128)
    # # for i in range(len(ys_waveguide)):
    # #     Ez_waveguide[i] = eigenmode_waveguide.amplitude(
    # #                 mp.Vector3(x_waveguide, ys_waveguide[i], 0), mp.Ez
    # #             )
    # # plt.plot(ys_waveguide, np.abs(Ez_waveguide))
    # # plt.xlabel('y (um)')
    # # plt.ylabel('Ez (a.u.)')
    # # plt.savefig('waveguide.png')
    # # plt.figure()
    # # Ez_fiber = np.zeros(len(xs_fiber), dtype=np.complex128)
    # # for i in range(len(xs_fiber)):
    # #     Ez_fiber[i] = eigenmode_fiber.amplitude(
    # #                 mp.Vector3(xs_fiber[i], y_fiber, 0), mp.Ez
    # #             )
    # # plt.plot(xs_fiber, np.abs(Ez_fiber))
    # plt.xlabel("x (um)")
    # plt.ylabel("Ez (a.u.)")
    # plt.savefig("fiber.png")
    # M1, E-field
    # plt.figure(figsize=(10, 8), dpi=100)
    # plt.suptitle(
    #     "MEEP get_eigenmode / MPB find_modes / Lumerical (manual)",
    #     y=1.05,
    #     fontsize=18,
    # )
    # plt.subplot(2, 2, 1)
    # mode_waveguide.plot_ez(show=False, operation=np.abs, scale=False)
    # plt.subplot(2, 2, 2)
    # mode_fiber.plot_ez(show=False, operation=np.abs, scale=False)
    # plt.subplot(2, 2, 3)
    # mode_waveguide.plot_hz(show=False, operation=np.abs, scale=False)
    # plt.subplot(2, 2, 4)
    # mode_fiber.plot_hz(show=False, operation=np.abs, scale=False)
    # plt.tight_layout()
    # plt.show()
    # fire.Fire(fiber)

    sim_dict = get_simulation_grating_farfield(fiber_xposition=1, fiber_angle_deg=15)
    plot(sim_dict["sim"])
    plt.show()
