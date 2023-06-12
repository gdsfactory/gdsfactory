"""Adapted from https://github.com/simbilod/option.

SMF specs from photonics.byu.edu/FiberOpticConnectors.parts/images/smf28.pdf

MFD:

- 10.4 for Cband
- 9.2 for Oband

"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import meep as mp
import numpy as np

from gdsfactory.serialization import clean_value_name
from gdsfactory.typings import Floats

nm = 1e-3
nSi = 3.47
nSiO2 = 1.44


def fiber_core_material(fiber_numerical_aperture, fiber_clad_material):
    return (fiber_numerical_aperture**2 + fiber_clad_material**2) ** 0.5


def get_simulation_grating_fiber(
    period: float = 0.66,
    fill_factor: float = 0.5,
    n_periods: int = 30,
    widths: Optional[Floats] = None,
    gaps: Optional[Floats] = None,
    fiber_angle_deg: float = 20.0,
    fiber_xposition: float = 1.0,
    fiber_core_diameter: float = 10.4,
    fiber_numerical_aperture: float = 0.14,
    fiber_clad_material: float = nSiO2,
    nwg: float = nSi,
    nslab: Optional[float] = None,
    clad_material: float = nSiO2,
    nbox: float = nSiO2,
    nsubstrate: float = nSi,
    pml_thickness: float = 1.0,
    substrate_thickness: float = 1.0,
    box_thickness: float = 2.0,
    core_thickness: float = 220 * nm,
    slab_thickness: float = 150 * nm,
    top_clad_thickness: float = 2.0,
    air_gap_thickness: float = 1.0,
    fiber_thickness: float = 2.0,
    resolution: int = 64,  # pixels/um
    wavelength_start: float = 1.4,
    wavelength_stop: float = 1.7,
    wavelength_points: int = 150,
    decay_by: float = 1e-3,
    eps_averaging: bool = False,
    fiber_port_y_offset_from_air: float = 1,
    waveguide_port_x_offset_from_grating_start: float = 10,
    fiber_port_x_size: Optional[float] = None,
    xmargin: float = 10.0,
) -> Dict[str, Any]:
    r"""Returns simulation results from grating coupler with fiber.

    na**2 = core_material**2 - clad_material**2
    core_material = sqrt(na**2 + core_material**2)

    Args:
        period: fiber grating period in um.
        fill_factor: fraction of the grating period filled with the grating material.
        n_periods: number of periods.
        widths: Optional list of widths. Overrides period, fill_factor, n_periods.
        gaps: Optional list of gaps. Overrides period, fill_factor, n_periods.
        fiber_angle_deg: fiber angle in degrees.
        fiber_xposition: xposition.
        fiber_core_diameter: fiber diameter. 10.4 for Cband and 9.2um for Oband.
        fiber_numerical_aperture: NA.
        fiber_clad_material: fiber cladding index.
        nwg: waveguide index.
        nslab: slab refractive index.
        clad_material: top cladding index.
        nbox: box index bottom.
        nsubstrate: index substrate.
        pml_thickness: pml_thickness (um).
        substrate_thickness: substrate_thickness (um).
        box_thickness: thickness for bottom cladding (um).
        core_thickness: core_thickness (um).
        slab_thickness: slab thickness (um). etch_depth=core_thickness-slab_thickness.
        top_clad_thickness: thickness of the top cladding.
        air_gap_thickness: air gap thickness.
        fiber_thickness: fiber_thickness.
        resolution: resolution pixels/um.
        wavelength_start: min wavelength (um).
        wavelength_stop: max wavelength (um).
        wavelength_points: wavelength points.
        eps_averaging: epsilon averaging.
        fiber_port_y_offset_from_air: y_offset from fiber to air (um).
        waveguide_port_x_offset_from_grating_start: in um.
        fiber_port_x_size: in um.
        xmargin: margin from PML to grating end in um.


    .. code::

                fiber_xposition
                     |
                fiber_core_diameter
             /     /  /     /       |
            /     /  /     /        | fiber_thickness
           /     /  /     /    _ _ _| _ _ _ _ _ _  _
                                    |
                                    | air_gap_thickness
                               _ _ _| _ _ _ _ _ _  _
                                    |
                   clad_material            | top_clad_thickness
                _   _   _      _ _ _| _ _ _ _ _ _  _
          nwg _| |_| |_| |__________|              _
                                    |               |
                 nslab              |core_thickness   | slab_thickness
                ______________ _ _ _|_ _ _ _ _ _ _ _|
                                    |
                 nbox               |box_thickness
                ______________ _ _ _|_ _ _ _ _ _ _ _
                                    |
                 nsubstrate         |substrate_thickness
                ______________ _ _ _|

    |--------------------|<-------->
                            xmargin

    """
    wavelengths = np.linspace(wavelength_start, wavelength_stop, wavelength_points)
    wavelength = np.mean(wavelengths)
    freqs = 1 / wavelengths
    widths = widths or n_periods * [period * fill_factor]
    gaps = gaps or n_periods * [period * (1 - fill_factor)]
    nslab = nslab or nwg

    settings = dict(
        widths=widths,
        gaps=gaps,
        n_periods=n_periods,
        nslab=nslab,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xposition=fiber_xposition,
        fiber_core_diameter=fiber_core_diameter,
        fiber_numerical_aperture=fiber_numerical_aperture,
        fiber_clad_material=fiber_clad_material,
        nwg=nwg,
        clad_material=clad_material,
        nbox=nbox,
        nsubstrate=nsubstrate,
        pml_thickness=pml_thickness,
        substrate_thickness=substrate_thickness,
        box_thickness=box_thickness,
        core_thickness=core_thickness,
        top_clad_thickness=top_clad_thickness,
        air_gap_thickness=air_gap_thickness,
        fiber_thickness=fiber_thickness,
        resolution=resolution,
        wavelength_start=wavelength_start,
        wavelength_stop=wavelength_stop,
        wavelength_points=wavelength_points,
        decay_by=decay_by,
        eps_averaging=eps_averaging,
        fiber_port_y_offset_from_air=fiber_port_y_offset_from_air,
        waveguide_port_x_offset_from_grating_start=waveguide_port_x_offset_from_grating_start,
        fiber_port_x_size=fiber_port_x_size,
    )
    settings_string = clean_value_name(settings)
    settings_hash = hashlib.md5(settings_string.encode()).hexdigest()[:8]

    # Angle in radians
    fiber_angle = np.radians(fiber_angle_deg)

    # Z (Y)-domain
    sz = (
        +pml_thickness
        + substrate_thickness
        + box_thickness
        + core_thickness
        + top_clad_thickness
        + air_gap_thickness
        + fiber_thickness
        + pml_thickness
    )
    # XY (X)-domain
    # Assume fiber port dominates
    fiber_port_y = (
        -sz / 2
        + core_thickness
        + top_clad_thickness
        + air_gap_thickness
        + fiber_port_y_offset_from_air
    )
    fiber_port_x_offset_from_angle = np.abs(fiber_port_y * np.tan(fiber_angle))
    length_grating = np.sum(widths) + np.sum(gaps)
    sxy = (
        2 * xmargin
        + 2 * pml_thickness
        + 2 * fiber_port_x_offset_from_angle
        + length_grating
    )

    # Materials from indices
    slab_material = mp.Medium(index=nslab)
    wg_material = mp.Medium(index=nwg)
    top_clad_material = mp.Medium(index=clad_material)
    bottom_clad_material = mp.Medium(index=nbox)
    fiber_core_material = (
        fiber_numerical_aperture**2 + fiber_clad_material**2
    ) ** 0.5
    fiber_clad_material = mp.Medium(index=fiber_clad_material)
    fiber_core_material = mp.Medium(index=fiber_core_material)

    # Useful reference point
    grating_start = (
        -fiber_xposition
    )  # Since fiber dominates, keep it centered and offset the grating

    # Initialize domain x-z plane simulation
    cell_size = mp.Vector3(sxy, sz)

    # Ports (position, sizes, directions)
    fiber_port_y = -sz / 2 + (
        +pml_thickness
        + substrate_thickness
        + box_thickness
        + core_thickness
        + top_clad_thickness
        + air_gap_thickness
        + fiber_port_y_offset_from_air
    )
    fiber_port_center = mp.Vector3(fiber_port_x_offset_from_angle, fiber_port_y)
    fiber_port_x_size = fiber_port_x_size or 3.5 * fiber_core_diameter
    fiber_port_size = mp.Vector3(fiber_port_x_size, 0, 0)
    # fiber_port_direction = mp.Vector3(y=-1).rotate(mp.Vector3(z=1), -1 * fiber_angle)

    waveguide_port_y = -sz / 2 + (
        +pml_thickness
        + substrate_thickness
        + box_thickness / 2
        + core_thickness / 2
        + top_clad_thickness / 2
    )
    waveguide_port_x = grating_start - waveguide_port_x_offset_from_grating_start
    waveguide_port_center = mp.Vector3(
        waveguide_port_x, waveguide_port_y
    )  # grating_start - dtaper, 0)
    waveguide_port_size = mp.Vector3(
        0, box_thickness + core_thickness / 2 + top_clad_thickness
    )
    waveguide_port_direction = mp.X

    # Geometry
    fiber_clad = 120
    hfiber_geom = 200  # Some large number to make fiber extend into PML

    geometry = [
        mp.Block(
            material=fiber_clad_material,
            center=mp.Vector3(0, waveguide_port_y - core_thickness / 2),
            size=mp.Vector3(fiber_clad, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    ]

    geometry.append(
        mp.Block(
            material=fiber_core_material,
            center=mp.Vector3(x=0),
            size=mp.Vector3(fiber_core_diameter, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    )

    # Air gap
    geometry.append(
        mp.Block(
            material=mp.air,
            center=mp.Vector3(
                0,
                -sz / 2
                + (
                    +pml_thickness
                    + substrate_thickness
                    + box_thickness
                    + core_thickness
                    + top_clad_thickness
                    + air_gap_thickness / 2
                ),
            ),
            size=mp.Vector3(mp.inf, air_gap_thickness),
        )
    )
    # Top cladding
    geometry.append(
        mp.Block(
            material=top_clad_material,
            center=mp.Vector3(
                0,
                -sz / 2
                + (
                    +pml_thickness
                    + substrate_thickness
                    + box_thickness
                    + core_thickness / 2
                    + top_clad_thickness / 2
                ),
            ),
            size=mp.Vector3(mp.inf, core_thickness + top_clad_thickness),
        )
    )
    # Bottom cladding
    geometry.append(
        mp.Block(
            material=bottom_clad_material,
            center=mp.Vector3(
                0,
                -sz / 2 + (+pml_thickness + substrate_thickness + box_thickness / 2),
            ),
            size=mp.Vector3(mp.inf, box_thickness),
        )
    )

    # slab
    geometry.append(
        mp.Block(
            material=slab_material,
            center=mp.Vector3(
                0,
                -sz / 2
                + (
                    +pml_thickness
                    + substrate_thickness
                    + box_thickness
                    + slab_thickness / 2
                ),
            ),
            size=mp.Vector3(mp.inf, slab_thickness),
        )
    )

    etch_depth = core_thickness - slab_thickness
    x = grating_start

    # grating teeth
    for width, gap in zip(widths, gaps):
        geometry.append(
            mp.Block(
                material=wg_material,
                center=mp.Vector3(
                    x + gap / 2,
                    -sz / 2
                    + (
                        +pml_thickness
                        + substrate_thickness
                        + box_thickness
                        + core_thickness
                        - etch_depth / 2
                    ),
                ),
                size=mp.Vector3(width, etch_depth),
            )
        )
        x += width + gap

    # waveguide
    geometry.append(
        mp.Block(
            material=wg_material,
            center=mp.Vector3(
                -sxy / 2,
                -sz / 2
                + (
                    +pml_thickness
                    + substrate_thickness
                    + box_thickness
                    + core_thickness
                    - etch_depth / 2
                ),
            ),
            size=mp.Vector3(sxy, etch_depth),
        )
    )

    # Substrate
    geometry.append(
        mp.Block(
            material=mp.Medium(index=nsubstrate),
            center=mp.Vector3(0, -sz / 2 + pml_thickness / 2 + substrate_thickness / 2),
            size=mp.Vector3(mp.inf, pml_thickness + substrate_thickness),
        )
    )

    # PMLs
    boundary_layers = [mp.PML(pml_thickness)]

    # mode frequency
    fcen = 1 / wavelength
    fwidth = 0.2 * fcen

    # Waveguide source
    sources_directions = [mp.X]
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
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
        eps_averaging=eps_averaging,
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
        settings_hash=settings_hash,
    )


def get_port_1D_eigenmode(
    sim_dict,
    band_num: int = 1,
    fiber_angle_deg: float = 15.0,
):
    """Args are the following.

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


def plot(sim, eps_parameters=None) -> None:
    """sim: simulation object."""
    sim.plot2D(eps_parameters=eps_parameters)
    # plt.colorbar()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Plotting
    epsilons = [1, 1.43482, 1.44, 1.44427, 3.47]

    eps_parameters = {"contour": True, "levels": np.unique(epsilons)}
    fiber_na = float(np.sqrt(1.44427**2 - 1.43482**2))

    sim_dict = get_simulation_grating_fiber(
        # grating parameters
        period=0.66,
        fill_factor=0.5,
        n_periods=30,
        # fiber parameters,
        fiber_angle_deg=20.0,
        fiber_xposition=0.0,
        fiber_core_diameter=9,
        fiber_numerical_aperture=fiber_na,
        fiber_clad_material=nSiO2,
        # material parameters
        nwg=3.47,
        clad_material=1.44,
        nbox=1.44,
        nsubstrate=3.47,
        # stack parameters
        pml_thickness=1.0,
        substrate_thickness=1.0,
        box_thickness=2.0,
        core_thickness=220 * nm,
        top_clad_thickness=2.0,
        air_gap_thickness=1.0,
        fiber_thickness=2.0,
        # simulation parameters
        resolution=50,
    )
    plot(sim_dict["sim"], eps_parameters=eps_parameters)
    # plot(sim_dict["sim"])
    plt.show()
