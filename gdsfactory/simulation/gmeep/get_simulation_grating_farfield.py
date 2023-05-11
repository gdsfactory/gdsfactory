"""FIXME: needs some work.

- figure out get_farfield outputs
- add tutorial in docs/notebooks/plugins/meep/002_gratings.ipynb
- add filecache
- benchmark with lumerical and tidy3d
- add tests

"""
from __future__ import annotations

from typing import Any, Dict, Optional

import meep as mp
import numpy as np

from gdsfactory.typings import Floats

nm = 1e-3
nSi = 3.48
nSiO2 = 1.44


def fiber_core_material(fiber_numerical_aperture, fiber_clad_material):
    return (fiber_numerical_aperture**2 + fiber_clad_material**2) ** 0.5


def get_simulation_grating_farfield(
    period: float = 0.66,
    fill_factor: float = 0.5,
    n_periods: int = 30,
    widths: Optional[Floats] = None,
    gaps: Optional[Floats] = None,
    etch_depth: float = 70 * nm,
    fiber_angle_deg: float = 20.0,
    fiber_xposition: float = 1.0,
    fiber_core_diameter: float = 10.4,
    fiber_numerical_aperture: float = 0.14,
    fiber_clad_material: float = nSiO2,
    core_material: float = nSi,
    clad_material: float = nSiO2,
    nsubstrate: float = nSi,
    pml_thickness: float = 1,
    box_thickness: float = 2.0,
    clad_thickness: float = 2.0,
    core_thickness: float = 220 * nm,
    resolution: int = 64,  # pixels/um
    wavelength_min: float = 1.5,
    wavelength_max: float = 1.6,
    wavelength_points: int = 50,
) -> Dict[str, Any]:
    """Returns grating coupler far field simulation.

    FIXME! needs some more work.

    na**2 = core_material**2 - clad_material**2
    core_material = sqrt(na**2 + core_material**2)

    Args:
        period: fiber grating period.
        fill_factor: fraction of the grating period
            filled with the grating material.
        n_periods: number of periods
        widths: Optional list of widths.
            Overrides period, fill_factor, n_periods.
        gaps: Optional list of gaps. Overrides period, fill_factor, n_periods.
        etch_depth: grating etch depth.
        fiber_angle_deg: fiber angle in degrees.
        fiber_xposition: xposition.
        fiber_core_diameter: fiber diameter.
        fiber_numerical_aperture: NA.
        fiber_clad_material: fiber cladding index.
        core_material: fiber index core.
        clad_material: top cladding index.
        nbox: box index bottom.
        nsubstrate: index substrate.
        pml_thickness: pml_thickness (um).
        substrate_thickness: substrate_thickness (um).
        box_thickness: thickness for bottom cladding (um).
        core_thickness: core_thickness (um).
        top_clad_thickness: thickness of the top cladding.
        air_gap_thickness: air gap thickness.
        resolution: resolution pixels/um.
        wavelength_min: min wavelength (um).
        wavelength_max: max wavelength (um).
        wavelength_points: wavelength points.


    Some parameters are different from get_simulation_grating_fiber
        fiber_thickness: fiber_thickness.

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
        fiber_clad_material=fiber_clad_material,
        resolution=resolution,
        core_material=core_material,
        clad_material=clad_material,
        nsubstrate=nsubstrate,
        n_periods=n_periods,
        box_thickness=box_thickness,
        clad_thickness=clad_thickness,
        etch_depth=etch_depth,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_points=wavelength_points,
        widths=widths,
        gaps=gaps,
    )
    length_grating = np.sum(widths) + np.sum(gaps)

    substrate_thickness = 1.0
    hair = 4
    core_material = mp.Medium(index=core_material)
    clad_material = mp.Medium(index=clad_material)
    fiber_angle = np.radians(fiber_angle_deg)

    y_offset = 0
    x_offset = 0

    # Minimally-parametrized computational cell
    # Could be further optimized

    # X-domain
    dbufferx = 0.5
    if length_grating < 3 * fiber_core_diameter:
        sxy = 3 * fiber_core_diameter + 2 * dbufferx + 2 * pml_thickness
    else:  # Fiber probably to the left
        sxy = (
            3 / 2 * fiber_core_diameter
            + length_grating / 2
            + 2 * dbufferx
            + 2 * pml_thickness
        )

    # Useful reference points
    cell_edge_left = -sxy / 2 + dbufferx + pml_thickness
    grating_start = -fiber_xposition

    # Y-domain (using z notation from 3D legacy code)
    dbuffery = 0.5
    sz = (
        2 * dbuffery
        + box_thickness
        + core_thickness
        + hair
        + substrate_thickness
        + 2 * pml_thickness
    )

    # Initialize domain x-z plane simulation
    cell_size = mp.Vector3(sxy, sz)

    # Ports (position, sizes, directions)
    fiber_offset_from_angle = (clad_thickness + core_thickness) * np.tan(fiber_angle)
    fiber_port_center = mp.Vector3(
        (0.5 * sz - pml_thickness + y_offset - 1) * np.sin(fiber_angle)
        + cell_edge_left
        + 3 / 2 * fiber_core_diameter
        - fiber_offset_from_angle,
        0.5 * sz - pml_thickness + y_offset - 1,
    )
    fiber_port_size = mp.Vector3(3 * fiber_core_diameter, 0, 0)
    # fiber_port_direction = mp.Vector3(y=-1).rotate(mp.Vector3(z=1), -1 * fiber_angle)

    waveguide_port_center = mp.Vector3(-sxy / 4)
    waveguide_port_size = mp.Vector3(0, 2 * clad_thickness - 0.2)
    waveguide_port_direction = mp.X

    # Geometry
    fiber_clad = 120
    hfiber_geom = 100  # Some large number to make fiber extend into PML

    fiber_core_material = (
        fiber_numerical_aperture**2 + fiber_clad_material**2
    ) ** 0.5
    fiber_clad_material = mp.Medium(index=fiber_clad_material)
    fiber_core_material = mp.Medium(index=fiber_core_material)

    geometry = [
        mp.Block(
            material=fiber_clad_material,
            center=mp.Vector3(
                x=grating_start + fiber_xposition - fiber_offset_from_angle
            ),
            size=mp.Vector3(fiber_clad, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    ]

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
                -0.5 * (core_thickness + substrate_thickness + pml_thickness + dbuffery)
                - box_thickness,
            ),
            size=mp.Vector3(mp.inf, substrate_thickness + pml_thickness + dbuffery),
        )
    )

    # PMLs
    boundary_layers = [mp.PML(pml_thickness)]

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
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        geometry=geometry,
        sources=sources,
        dimensions=2,
        eps_averaging=True,
    )

    offset_vector = mp.Vector3(x_offset, y_offset)
    nearfield = sim.add_near2far(
        fcen,
        0,
        1,
        mp.Near2FarRegion(
            mp.Vector3(x_offset, 0.5 * sz - pml_thickness + y_offset) - offset_vector,
            size=mp.Vector3(sxy - 2 * pml_thickness, 0),
        ),
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
        nearfield=nearfield,
    )


def get_farfield(wavelength: float = 1.55, **kwargs):
    """FIXME: figure out outputs.

    based on
    http://www.simpetus.com/projects.html#meep_outcoupler
    """
    sim_dict = get_simulation_grating_farfield(**kwargs)

    sim = sim_dict["sim"]
    sim.run(until=400)

    fcen = 1 / wavelength
    r = 1000 / fcen  # 1000 wavelengths out from the source
    npts = 1000  # number of points in [0,2*pi) range of angles

    farfield_angles = []
    farfield_power = []

    nearfield = sim["nearfield"]
    for n in range(npts):
        ff = sim.get_farfield(
            nearfield,
            mp.Vector3(r * np.cos(np.pi * (n / npts)), r * np.sin(np.pi * (n / npts))),
        )
        farfield_angles.append(
            np.angle(np.cos(np.pi * (n / npts)) + 1j * np.sin(np.pi * (n / npts)))
        )
        farfield_power.append(ff)

    farfield_angles = np.array(farfield_angles)
    farfield_power = np.array(farfield_power)

    return sim.get_eigenmode_coefficients(
        sim_dict["waveguide_monitor"], [1], eig_parity=mp.ODD_Z, direction=mp.X
    )


def get_port_1D_eigenmode(
    sim_dict,
    band_num: int = 1,
    fiber_angle_deg: float = 15,
):
    """Args are the following.

        sim_dict: simulation dict
        band_num: band number to solve for
        fiber_angle_deg

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


def plot(sim) -> None:
    """sim: simulation object."""
    sim.plot2D(eps_parameters={"contour": True})
    # plt.colorbar()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim_dict = get_simulation_grating_farfield(fiber_xposition=1, fiber_angle_deg=15)

    # plot(sim_dict["sim"])
    # plt.show()

    # results = {}
    # for angle in [10]:
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

    # plt.xlabel("x (um)")
    # plt.ylabel("Ez (a.u.)")
    # plt.savefig("fiber.png")

    # # M1, E-field
    # plt.figure(figsize=(10, 8), dpi=100)
    # plt.suptitle(
    #     "MEEP get_eigenmode / MPB find_modes / Lumerical (manual)",
    #     y=1.05,
    #     fontsize=18,
    # )
    # plt.show()

    wavelength = 1.55
    settings = {}
    sim_dict = get_simulation_grating_farfield(**settings)

    sim = sim_dict["sim"]
    sim.run(until=100)
    # sim.run(until=400)

    fcen = 1 / wavelength
    r = 1000 / fcen  # 1000 wavelengths out from the source
    npts = 1000  # number of points in [0,2*pi) range of angles

    farfield_angles = []
    farfield_power = []

    nearfield = sim["nearfield"]
    for n in range(npts):
        ff = sim.get_farfield(
            nearfield,
            mp.Vector3(r * np.cos(np.pi * (n / npts)), r * np.sin(np.pi * (n / npts))),
        )
        farfield_angles.append(
            np.angle(np.cos(np.pi * (n / npts)) + 1j * np.sin(np.pi * (n / npts)))
        )
        farfield_power.append(ff)

    farfield_angles = np.array(farfield_angles)
    farfield_power = np.array(farfield_power)

    # Waveguide
    res_waveguide = sim.get_eigenmode_coefficients(
        sim_dict["waveguide_monitor"], [1], eig_parity=mp.ODD_Z, direction=mp.X
    )
    print(res_waveguide)
    plt.plot(farfield_power)
    plt.show()
