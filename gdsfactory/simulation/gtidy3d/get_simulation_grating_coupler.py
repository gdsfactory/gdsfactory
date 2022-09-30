"""Returns tidy3d simulation from gdsfactory Component."""
import warnings
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.extension import move_polar_rad_copy
from gdsfactory.config import logger
from gdsfactory.pdk import get_layer_stack
from gdsfactory.simulation.gtidy3d.materials import (
    MATERIAL_NAME_TO_TIDY3D_INDEX,
    MATERIAL_NAME_TO_TIDY3D_NAME,
    get_index,
    get_medium,
)
from gdsfactory.tech import LayerStack


def get_simulation_grating_coupler(
    component: Component,
    port_extension: Optional[float] = 15.0,
    layer_stack: Optional[LayerStack] = None,
    thickness_pml: float = 1.0,
    xmargin: float = 0,
    ymargin: float = 0,
    xmargin_left: float = 0,
    xmargin_right: float = 0,
    ymargin_top: float = 0,
    ymargin_bot: float = 0,
    zmargin: float = 1.0,
    clad_material: str = "sio2",
    box_material: str = "sio2",
    box_thickness: float = 2.0,
    substrate_material: str = "si",
    port_waveguide_name: str = "o1",
    port_margin: float = 0.5,
    port_waveguide_offset: float = 0.1,
    distance_source_to_monitors: float = 0.2,
    wavelength: Optional[float] = 1.55,
    wavelength_start: float = 1.20,
    wavelength_stop: float = 1.80,
    wavelength_points: int = 256,
    plot_modes: bool = False,
    num_modes: int = 2,
    run_time_ps: float = 10.0,
    fiber_port_name: str = "vertical_te",
    fiber_xoffset: float = 0,
    fiber_z: float = 2,
    fiber_mfd: float = 5.2,
    fiber_angle_deg: float = 20.0,
    dispersive: bool = False,
    material_name_to_tidy3d_index: Dict[str, float] = MATERIAL_NAME_TO_TIDY3D_INDEX,
    material_name_to_tidy3d_name: Dict[str, str] = MATERIAL_NAME_TO_TIDY3D_NAME,
    is_3d: bool = True,
    with_all_monitors: bool = False,
    boundary_spec: Optional[td.BoundarySpec] = None,
    grid_spec: Optional[td.GridSpec] = None,
    sidewall_angle_deg: float = 0,
    dilation: float = 0.0,
    **kwargs,
) -> td.Simulation:
    r"""Returns Simulation object from a gdsfactory grating coupler component.

    injects a Gaussian beam from above and monitors the transmission into the waveguide.

    based on grating coupler example
    https://docs.simulation.cloud/projects/tidy3d/en/latest/notebooks/GratingCoupler.html

    .. code::

         top view
              ________________________________
             |                               |
             | xmargin_left                  |
             |<------>                       |
             |           ________________    |
             |          /   |  |  |  |  |    |
             |         /    |  |  |  |  |    |
             |=========     |  |  |  |  |    |
             |         \    |  |  |  |  |    |
             |   _ _ _ _\___|__|__|__|__|    |
             |   |                       <-->|
             |   |ymargin_bot   xmargin_right|
             |   |                           |
             |___|___________________________|


        side view

              fiber_xoffset
                 |<--->|
            fiber_port_name
                 |
                         fiber_angle_deg > 0
                        |  /
                        | /
                        |/
                 /              /       |
                /  fiber_mfd   /        |
               /<------------>/    _ _ _| _ _ _ _ _ _ _
                                        |
                   clad_material        | fiber_z
                    _   _   _      _ _ _|_ _ _ _ _ _ _
                   | | | | | |          ||wg_thickness
                  _| |_| |_| |__________||___
                                        || |
        waveguide            |          || | slab_thickness
              ____________________ _ _ _||_|_
                             |          |
                   box_material         |box_thickness
              _______________|____ _ _ _|_ _ _ _ _ _ _
                             |          |
                 substrate_material     |substrate_thickness
             ________________|____ _ _ _|_ _ _ _ _ _ _
                             |
        |--------------------|<-------->
                                xmargin

    Args:
        component: gdsfactory Component.
        port_extension: extend ports beyond the PML.
        layer_stack: contains layer to thickness, zmin and material.
            Defaults to active pdk.layer_stack.
        thickness_pml: PML thickness (um).
        xmargin: left/right distance from component to PML.
        xmargin_left: left distance from component to PML.
        xmargin_right: right distance from component to PML.
        ymargin: left/right distance from component to PML.
        ymargin_top: top distance from component to PML.
        ymargin_bot: bottom distance from component to PML.
        zmargin: thickness for cladding above and below core.
        clad_material: material for cladding.
        box_material:
        substrate_material:
        box_thickness: (um).
        substrate_thickness: (um).
        port_waveguide_name: input port name.
        port_margin: margin on each side of the port.
        distance_source_to_monitors: in (um) source goes before monitors.
        port_waveguide_offset: mode solver workaround.
            positive moves source forward, negative moves source backward.
        wavelength: source center wavelength (um).
            if None takes mean between wavelength_start, wavelength_stop
        wavelength_start: in (um).
        wavelength_stop: in (um).
        wavelength_points: number of wavelengths.
        plot_modes: plot source modes.
        num_modes: number of modes to plot.
        run_time_ps: make sure it's sufficient for the fields to decay.
            defaults to 10ps and automatic shutoff stops earlier if needed.
        fiber_port_name: for the component.
        fiber_xoffset: fiber center xoffset to fiber_port_name.
        fiber_z: fiber zoffset from grating zmax.
        fiber_mfd: fiber mode field diameter (um).
        fiber_angle_deg: fiber_angle in degrees with respect to normal.
            Positive for west facing, Negative for east facing sources.
        dispersive: False uses constant refractive index materials.
            True adds wavelength depending materials.
            Dispersive materials require more computation.
        material_name_to_tidy3d_index: not dispersive materials have a constant index.
        material_name_to_tidy3d_name: dispersive materials have a wavelength
            dependent index. Maps layer_stack names with tidy3d material database names.
        is_3d: if False collapses the Y direction for a 2D simulation.
        with_all_monitors: True includes field monitors which increase results filesize.
        grid_spec: defaults to automatic td.GridSpec.auto(wavelength=wavelength)
            td.GridSpec.uniform(dl=20*nm)
            td.GridSpec(
                grid_x = td.UniformGrid(dl=0.04),
                grid_y = td.AutoGrid(min_steps_per_wvl=20),
                grid_z = td.AutoGrid(min_steps_per_wvl=20),
                wavelength=wavelength,
                override_structures=[refine_box]
            )
        boundary_spec: Specification of boundary conditions along each dimension.
            Defaults to td.BoundarySpec.all_sides(boundary=td.PML())
        dilation: float = 0.0
            Dilation of the polygon in the base by shifting each edge along its
            normal outwards direction by a distance;
            a negative value corresponds to erosion.
        sidewall_angle_deg : float = 0
            Angle of the sidewall.
            ``sidewall_angle=0`` (default) specifies vertical wall,
            while ``0<sidewall_angle_deg<90`` for the base to be larger than the top.

    keyword Args:
        symmetry: Define Symmetries.
            Tuple of integers defining reflection symmetry across a plane
            bisecting the simulation domain normal to the x-, y-, and z-axis
            at the simulation center of each axis, respectvely.
            Each element can be ``0`` (no symmetry), ``1`` (even, i.e. 'PMC' symmetry) or
            ``-1`` (odd, i.e. 'PEC' symmetry).
            Note that the vectorial nature of the fields must be taken into account to correctly
            determine the symmetry value.
        medium: Background medium of simulation, defaults to vacuum if not specified.
        shutoff: shutoff condition
            Ratio of the instantaneous integrated E-field intensity to the maximum value
            at which the simulation will automatically terminate time stepping.
            Used to prevent extraneous run time of simulations with fully decayed fields.
            Set to ``0`` to disable this feature.
        subpixel: subpixel averaging.If ``True``, uses subpixel averaging of the permittivity
        based on structure definition, resulting in much higher accuracy for a given grid size.
        courant: courant factor.
            Courant stability factor, controls time step to spatial step ratio.
            Lower values lead to more stable simulations for dispersive materials,
            but result in longer simulation times.
        version: String specifying the front end version number.

    .. code::

        import matplotlib.pyplot as plt
        import gdsfactory as gf
        import gdsfactory.simulation.gtidy3d as gt

        c = gf.components.grating_coupler_elliptical_arbitrary(
            widths=[0.343] * 25, gaps=[0.345] * 25
        )
        sim = gt.get_simulation(c)
        gt.plot_simulation(sim)

    """
    layer_stack = layer_stack or get_layer_stack()

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_material = layer_stack.get_layer_to_material()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    # layer_to_sidewall_angle = layer_stack.get_layer_to_sidewall_angle()

    boundary_spec = boundary_spec or td.BoundarySpec.all_sides(boundary=td.PML())
    grid_spec = grid_spec or td.GridSpec.auto(wavelength=wavelength)

    if dispersive:
        material_name_to_tidy3d = material_name_to_tidy3d_name
    else:
        material_name_to_tidy3d = material_name_to_tidy3d_index

    assert isinstance(
        component, Component
    ), f"component needs to be a gf.Component, got Type {type(component)}"

    if port_waveguide_name not in component.ports:
        warnings.warn(
            f"port_waveguide_name={port_waveguide_name} not in {component.ports.keys()}"
        )
        port_waveguide = component.get_ports_list()[0]
        port_waveguide_name = port_waveguide.name
        warnings.warn(f"Selecting port_waveguide_name={port_waveguide_name} instead.")

    if fiber_port_name not in component.ports:
        raise ValueError(
            f"fiber_port_name = {fiber_port_name!r} not in {component.ports.keys()}"
        )

    component_padding = gf.add_padding_container(
        component,
        default=0,
        top=ymargin or ymargin_top,
        bottom=ymargin or ymargin_bot,
        left=xmargin or xmargin_left,
        right=xmargin or xmargin_right,
    )
    component_extended = (
        gf.components.extension.extend_ports(
            component=component_padding, length=port_extension, centered=True
        )
        if port_extension
        else component
    )

    gf.show(component_extended)
    component_extended = component_extended.flatten()
    component_ref = component_padding.ref()
    component_ref.x = 0
    component_ref.y = 0

    layers_thickness = [
        layer_to_thickness[layer]
        for layer in component.get_layers()
        if layer in layer_to_thickness
    ]

    if len(layer_to_thickness) < 1:
        raise ValueError(f"{component.get_layers()} not in {layer_to_thickness.keys()}")

    wg_thickness = max(layers_thickness)
    sim_xsize = component_ref.xsize + 2 * thickness_pml
    sim_zsize = (
        thickness_pml + box_thickness + wg_thickness + thickness_pml + 2 * zmargin
    )
    sim_ysize = component_ref.ysize + 2 * thickness_pml if is_3d else 0
    sim_size = [
        sim_xsize,
        sim_ysize,
        sim_zsize,
    ]

    clad_material_name_or_index = material_name_to_tidy3d[clad_material]
    box_material_name_or_index = material_name_to_tidy3d[box_material]
    substrate_material_name_or_index = material_name_to_tidy3d[substrate_material]

    clad = td.Structure(
        geometry=td.Box(
            size=(td.inf, td.inf, sim_zsize),
            center=(0, 0, sim_zsize / 2),
        ),
        medium=get_medium(name_or_index=clad_material_name_or_index),
    )
    box = td.Structure(
        geometry=td.Box(
            size=(td.inf, td.inf, box_thickness),
            center=(0, 0, -box_thickness / 2),
        ),
        medium=get_medium(name_or_index=box_material_name_or_index),
    )

    substrate_thickness = 10
    substrate = td.Structure(
        geometry=td.Box(
            size=(td.inf, td.inf, substrate_thickness),
            center=(0, 0, -box_thickness - substrate_thickness / 2),
        ),
        medium=get_medium(name_or_index=substrate_material_name_or_index),
    )

    structures = [substrate, box, clad]

    for layer in component.layers:
        if layer in layer_to_thickness and layer in layer_to_material:
            thickness = layer_to_thickness[layer]
            zmin = layer_to_zmin[layer]
            zmax = zmin + thickness
            if layer_to_material[layer] in material_name_to_tidy3d:
                name_or_index = material_name_to_tidy3d[layer_to_material[layer]]
                medium = get_medium(name_or_index=name_or_index)
                index = get_index(name_or_index=name_or_index)
                logger.debug(
                    f"Add {layer}, {name_or_index!r}, index = {index:.3f}, "
                    f"thickness = {thickness}, zmin = {zmin}, zmax = {zmax}"
                )

                polygons = td.PolySlab.from_gds(
                    gds_cell=component_extended,
                    gds_layer=layer[0],
                    gds_dtype=layer[1],
                    axis=2,
                    slab_bounds=(zmin, zmax),
                    sidewall_angle=np.deg2rad(sidewall_angle_deg),
                    dilation=dilation,
                )

                for polygon in polygons:
                    geometry = td.Structure(
                        geometry=polygon,
                        medium=medium,
                    )
                    structures.append(geometry)
            elif layer not in layer_to_material:
                logger.debug(f"Layer {layer} not in {layer_to_material.keys()}")
            else:
                materials = list(material_name_to_tidy3d.keys())
                logger.debug(f"material {layer_to_material[layer]} not in {materials}")

    wavelengths = np.linspace(wavelength_start, wavelength_stop, wavelength_points)
    freqs = td.constants.C_0 / wavelengths
    freq0 = td.constants.C_0 / np.mean(wavelengths)
    fwidth = freq0 / 10

    # Add input waveguide port
    port = component_ref.ports[port_waveguide_name]
    angle = port.orientation
    width = port.width + 2 * port_margin
    size_x = width * abs(np.sin(angle * np.pi / 180))
    size_y = width * abs(np.cos(angle * np.pi / 180))
    size_x = 0 if size_x < 0.001 else size_x
    size_y = 0 if size_y < 0.001 else size_y
    size_y = size_y if is_3d else td.inf
    size_z = wg_thickness + 2 * zmargin
    waveguide_port_size = [size_x, size_y, size_z]
    xy_shifted = move_polar_rad_copy(
        np.array(port.center), angle=angle * np.pi / 180, length=port_waveguide_offset
    )
    waveguide_port_center = xy_shifted.tolist() + [0]  # (x, y, z=0)

    waveguide_monitor = td.ModeMonitor(
        center=waveguide_port_center,
        size=waveguide_port_size,
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=1),
        name="waveguide",
    )

    # Add fiber monitor
    fiber_port = component_ref.ports[fiber_port_name]
    fiber_port_x = fiber_port.x + fiber_xoffset

    assert -sim_size[0] / 2 < fiber_port_x < sim_size[0] / 2, (
        f"component.ports[{fiber_port_name!r}] + (fiber_xoffset = {fiber_xoffset}). "
        f"{fiber_port_x} needs to be between {-sim_size[0]/2} and {+sim_size[0]/2}"
    )

    # inject Gaussian beam from above and monitors the transmission into the waveguide.
    gaussian_beam = td.GaussianBeam(
        size=(td.inf, td.inf, 0),
        center=[fiber_port_x, 0, fiber_z],
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        angle_theta=np.deg2rad(-fiber_angle_deg),
        angle_phi=np.pi,
        direction="-",
        waist_radius=fiber_mfd / 2,
        pol_angle=np.pi / 2,
    )

    plane_monitor = td.FieldMonitor(
        center=[0, 0, (zmax + zmin) / 2],
        size=[sim_size[0], sim_size[1], 0],
        freqs=[freq0],
        name="full_domain_fields",
    )

    rad_monitor = td.FieldMonitor(
        center=[0, 0, 0],
        size=[td.inf, 0, td.inf],
        freqs=[freq0],
        name="radiated_fields",
    )

    near_field_monitor = td.FieldMonitor(
        center=[0, 0, fiber_z],
        size=[td.inf, td.inf, 0],
        freqs=[freq0],
        name="radiated_near_fields",
    )

    monitors = [waveguide_monitor]
    monitors += (
        [plane_monitor, rad_monitor, near_field_monitor] if with_all_monitors else []
    )

    sim = td.Simulation(
        size=sim_size,
        structures=structures,
        sources=[gaussian_beam],
        monitors=monitors,
        run_time=run_time_ps * 1e-12,
        boundary_spec=boundary_spec,
        **kwargs,
    )

    if plot_modes:
        src_plane = td.Box(center=waveguide_port_center, size=waveguide_port_size)
        ms = td.plugins.ModeSolver(simulation=sim, plane=src_plane, freq=freq0)

        mode_spec = td.ModeSpec(num_modes=num_modes)
        modes = ms.solve(mode_spec=mode_spec)

        print(
            "Effective index of computed modes: ",
            ", ".join([f"{mode.n_eff:1.4f}" for mode in modes]),
        )

        if is_3d:
            fig, axs = plt.subplots(num_modes, 2, figsize=(12, 12))
        else:
            fig, axs = plt.subplots(num_modes, 3, figsize=(12, 12))

        for mode_ind in range(num_modes):
            if is_3d:
                abs(modes[mode_ind].field_data.Ey).plot(
                    x="y", y="z", cmap="magma", ax=axs[mode_ind, 0]
                )
                abs(modes[mode_ind].field_data.Ez).plot(
                    x="y", y="z", cmap="magma", ax=axs[mode_ind, 1]
                )
            else:
                abs(modes[mode_ind].field_data.Ex).plot(ax=axs[mode_ind, 0])
                abs(modes[mode_ind].field_data.Ey).plot(ax=axs[mode_ind, 1])
                abs(modes[mode_ind].field_data.Ez).plot(ax=axs[mode_ind, 2])

                axs[mode_ind, 0].set_title(f"|Ex|: mode_index={mode_ind}")
                axs[mode_ind, 1].set_title(f"|Ey|: mode_index={mode_ind}")
                axs[mode_ind, 2].set_title(f"|Ez|: mode_index={mode_ind}")

        if is_3d:
            axs[mode_ind, 0].set_aspect("equal")
            axs[mode_ind, 1].set_aspect("equal")
        plt.show()
    return sim


if __name__ == "__main__":
    import gdsfactory.simulation.gtidy3d as gt

    c = gf.components.grating_coupler_elliptical_arbitrary(
        widths=[0.343] * 25, gaps=[0.345] * 25
    )
    sim = get_simulation_grating_coupler(
        c,
        plot_modes=False,
        is_3d=False,
        fiber_angle_deg=20,
    )
    gt.plot_simulation(sim)  # make sure simulations looks good

    # c = gf.components.grating_coupler_elliptical_lumerical()  # inverse design grating
    # sim = get_simulation_grating_coupler(c, plot_modes=False, fiber_angle_deg=-5)
    # sim_data = gt.get_results(sim).result()
    # freq0 = td.constants.C_0 / 1.55
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, tight_layout=True, figsize=(14, 16))
    # sim_data.plot_field("full_domain_fields", "Ey", freq=freq0, z=0, ax=ax1)
    # sim_data.plot_field("radiated_near_fields", "Ey", freq=freq0, z=0, ax=ax2)
    # sim_data.plot_field("radiated_fields", "Ey", freq=freq0, y=0, ax=ax3)

    # plt.figure()
    # waveguide = sim_data["waveguide"]
    # plt.plot(np.abs(waveguide.amps.values[0].flatten()), label="+")
    # plt.plot(np.abs(waveguide.amps.values[1].flatten()), label="-")
    # plt.legend()
    # plt.show()
    # print(f"waveguide in waveguide / waveguide in = {float(waveguide.amps.values):.2f} ")
