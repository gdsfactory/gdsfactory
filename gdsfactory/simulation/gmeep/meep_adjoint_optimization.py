from types import LambdaType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import nlopt
import numpy as np
from meep import Block, EigenModeSource, MaterialGrid, Simulation, Vector3, Volume
from meep.adjoint import DesignRegion, EigenmodeCoefficient, OptimizationProblem
from meep.visualization import get_2D_dimensions
from numpy import ndarray

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.simulation.gmeep import get_simulation
from gdsfactory.tech import LayerStack
from gdsfactory.types import Layer


def get_meep_adjoint_optimizer(
    component: Component,
    objective_function: Callable,
    design_regions: List[DesignRegion],
    design_variables: List[MaterialGrid],
    design_update: np.ndarray,
    TE_mode_number: int = 1,
    resolution: int = 30,
    cell_size: Optional[Tuple] = None,
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
):
    """Return a Meep `OptimizationProblem` object.

    Args:
        component: gdsfactory component.
        objective_function: functions must be composed of "field functions" that transform the recorded fields.
        design_regions: list of DesignRegion objects.
        design_variables: list of MaterialGrid objects.
        design_update: ndarray to intializethe optimization.
        TE_mode_number: TE mode number.
        resolution: in pixels/um (20: for coarse, 120: for fine).
        cell_size: tuple of Simulation object dimensions in um.
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
        opt: OptimizationProblem object
    """
    sim_dict = get_simulation(
        component,
        resolution=resolution,
        extend_ports_length=extend_ports_length,
        layer_stack=layer_stack,
        zmargin_top=zmargin_top,
        zmargin_bot=zmargin_bot,
        tpml=tpml,
        clad_material=clad_material,
        is_3d=is_3d,
        wavelength_start=wavelength_start,
        wavelength_stop=wavelength_stop,
        wavelength_points=wavelength_points,
        dfcen=dfcen,
        port_source_name=port_source_name,
        port_margin=port_margin,
        distance_source_to_monitors=distance_source_to_monitors,
        port_source_offset=port_source_offset,
        port_monitor_offset=port_monitor_offset,
        dispersive=dispersive,
        material_name_to_meep=material_name_to_meep,
        **settings,
    )
    sim = sim_dict["sim"]

    design_regions_geoms = [
        Block(
            center=design_region.center,
            size=design_region.size,
            material=design_variable,
        )
        for design_region, design_variable in zip(design_regions, design_variables)
    ]

    for design_region_geom in design_regions_geoms:
        sim.geometry.append(design_region_geom)

    cell_thickness = sim.cell_size[2]

    monitors = sim_dict["monitors"]
    ob_list = [
        EigenmodeCoefficient(
            sim,
            Volume(
                center=monitor.regions[0].center,
                size=monitor.regions[0].size,
            ),
            TE_mode_number,
        )
        for monitor in monitors.values()
    ]

    c = component.copy()
    for design_region, design_variable in zip(design_regions, design_variables):
        sim.geometry.append(
            Block(design_region.size, design_region.center, material=design_variable)
        )
        block = c << gf.components.rectangle(
            (design_region.size[0], design_region.size[1])
        )
        block.center = (design_region.center[0], design_region.center[1])

    sim.cell_size = (
        Vector3(*cell_size)
        if cell_size
        else Vector3(
            c.xsize + 2 * sim.boundary_layers[0].thickness,
            c.ysize + 2 * sim.boundary_layers[0].thickness,
            cell_thickness,
        )
    )

    source = [
        EigenModeSource(
            sim.sources[0].src,
            eig_band=1,
            direction=sim.sources[0].direction,
            eig_kpoint=Vector3(1, 0, 0),
            size=sim.sources[0].size,
            center=sim.sources[0].center,
        )
    ]

    sim.sources = source

    opt = OptimizationProblem(
        simulation=sim,
        objective_functions=[objective_function],
        objective_arguments=ob_list,
        design_regions=design_regions,
        frequencies=sim_dict["freqs"],
        decay_by=settings.get("decay_by", 1e-5),
    )

    opt.update_design([design_update])
    opt.plot2D(True)

    return opt


def run_meep_adjoint_optimizer(
    number_of_params: int,
    cost_function: LambdaType,
    update_variable: np.ndarray,
    maximize_cost_function: bool = True,
    algorithm: int = nlopt.LD_MMA,
    lower_bound: Any = 0,
    upper_bound: Any = 1,
    maxeval: int = 10,
    get_optimized_component: bool = False,
    opt: OptimizationProblem = None,
    **kwargs,
) -> Union[ndarray, Component]:
    """Run adjoint optimization using Meep.

    Args:
        number_of_params: number of parameters to optimize (usually resolution_in_x * resolution_in_y).
        cost_function: cost function to optimize.
        update_variable: variable to update the optimization with.
        maximize_cost_function: if True, maximize the cost function, else minimize it.
        algorithm: nlopt algorithm to use (default: nlopt.LD_MMA).
        lower_bound: lower bound for the optimization.
        upper_bound: upper bound for the optimization.
        maxeval: maximum number of evaluations.
        get_optimized_component: if True, returns the optimized gdsfactory Component.
            If this is True, the O  ptimization object used for the optimization must be passed as an argument.
        opt: OptimizationProblem object used for the optimization. Used only if get_optimized_component is True.

    Keyword Args:
        fcen: center frequency of the source.
        upscale_factor: upscale factor for the optimization's grid.
        threshold_offset_from_max: threshold offset from max eps value.
        layer: layer to apply to the optimized component.
    """
    solver = nlopt.opt(algorithm, number_of_params)
    solver.set_lower_bounds(lower_bound)
    solver.set_upper_bounds(upper_bound)
    if maximize_cost_function:
        solver.set_max_objective(cost_function)
    else:
        solver.set_min_objective(cost_function)
    solver.set_maxeval(maxeval)
    update_variable[:] = solver.optimize(update_variable)

    if get_optimized_component:
        fcen = kwargs.get("fcen", 1 / 1.55)
        upscale_factor = kwargs.get("upscale_factor", 2)
        threshold_offset_from_max = kwargs.get("threshold_offset_from_max", 0.01)
        layer = kwargs.get("layer", (1, 0))

        return get_component_from_sim(
            opt.sim, fcen, upscale_factor, threshold_offset_from_max, layer
        )
    return update_variable


def get_component_from_sim(
    sim: Simulation,
    fcen: float = 1 / 1.55,
    upscale_factor: int = 2,
    threshold_offset_from_max: float = 2.0,
    layer: Layer = (1, 0),
) -> Component:
    """Get gdsfactory Component from Meep Simulation object.

    Args:
        sim: Meep Simulation object.
        fcen: center frequency of the source.
        upscale_factor: upscale factor for the optimization's grid.
        threshold_offset_from_max: threshold offset from max eps value.
        layer: layer to apply to the optimized component.

    Returns:
        gdsfactory Component.
    """
    grid_resolution = upscale_factor * sim.resolution
    sim_center, sim_size = get_2D_dimensions(sim, output_plane=None)
    xmin = sim_center.x - sim_size.x / 2
    xmax = sim_center.x + sim_size.x / 2
    ymin = sim_center.y - sim_size.y / 2
    ymax = sim_center.y + sim_size.y / 2
    Nx = int((xmax - xmin) * grid_resolution + 1)
    Ny = int((ymax - ymin) * grid_resolution + 1)
    xtics = np.linspace(xmin, xmax, Nx)
    ytics = np.linspace(ymin, ymax, Ny)
    ztics = np.array([sim_center.z])
    eps_data = np.real(sim.get_epsilon_grid(xtics, ytics, ztics, frequency=fcen))
    return gf.read.from_np(
        eps_data,
        nm_per_pixel=1e3 / grid_resolution,
        layer=layer,
        threshold=np.max(eps_data) - threshold_offset_from_max,
    )


def _example_optim_geometry() -> Component:
    """Dummy example of a component to optimize."""
    from meep import Medium

    design_region_width = 5
    design_region_height = 4

    resolution = 20
    design_region_resolution = int(5 * resolution)

    Nx = int(design_region_resolution * design_region_width)
    Ny = int(design_region_resolution * design_region_height)

    pml_size = 1.0
    waveguide_length = 0.5
    Sx = 2 * pml_size + 2 * waveguide_length + design_region_width

    SiO2 = Medium(index=1.44)
    Si = Medium(index=3.4)

    design_variables = MaterialGrid(Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN")
    design_region = DesignRegion(
        design_variables,
        volume=Volume(
            center=Vector3(),
            size=Vector3(design_region_width, design_region_height, 0),
        ),
    )

    c = Component("mmi1x2")

    arm_separation = 1.0
    straight1 = c << gf.components.straight(Sx / 2 + 1)
    straight1.move(straight1.ports["o2"], (-design_region_width / 2.0, 0))
    straight2 = c << gf.components.straight(Sx / 2 + 1)
    straight2.move(
        straight2.ports["o1"], (design_region_width / 2.0, (arm_separation + 1.0) / 2.0)
    )
    straight3 = c << gf.components.straight(Sx / 2 + 1)
    straight3.move(
        straight3.ports["o1"],
        (design_region_width / 2.0, (-arm_separation - 1.0) / 2.0),
    )

    c.add_port("o1", port=straight1.ports["o1"])
    c.add_port("o2", port=straight2.ports["o2"])
    c.add_port("o3", port=straight3.ports["o2"])

    return design_region, design_variables, c, Nx, Ny


if __name__ == "__main__":
    import autograd.numpy as npa

    eta_i = 0.5

    design_region, design_variables, c, Nx, Ny = _example_optim_geometry()

    seed = 240
    np.random.seed(seed)
    x0 = np.random.rand(
        Nx * Ny,
    )

    def J(source, top, bottom):
        power = npa.abs(top / source) ** 2 + npa.abs(bottom / source) ** 2
        return npa.mean(power)

    opt = get_meep_adjoint_optimizer(
        c,
        J,
        [design_region],
        [design_variables],
        x0,
        cell_size=(15, 8),
        extend_ports_length=0,
        port_margin=0.75,
        port_source_offset=-3.5,
        port_monitor_offset=-3.5,
    )

    opt.plot2D(True)
