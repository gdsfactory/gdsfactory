import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.simulation.gmeep import get_meep_adjoint_optimizer
from gdsfactory.simulation.gmeep.get_meep_geometry import get_meep_geometry_from_component

from meep import EigenModeSource, GaussianSource, Medium, Mirror, NO_DIRECTION, PML, Simulation, Vector3, Y


def _example_optim_geometry() -> Component:
    """Dummy example of a component to optimize."""

    design_region_width = 5

    pml_size = 1.0
    waveguide_length = 0.5
    Sx = 2 * pml_size + 2 * waveguide_length + design_region_width

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

    return c


def _get_sim_from_meep() -> Simulation:
    c= _example_optim_geometry()

    geometry = get_meep_geometry_from_component(c)

    design_region_width = 2.5
    design_region_height = 2.5
    waveguide_length = 0.5
    pml_size = 1.0
    resolution = 20

    Sx = 2 * pml_size + 2 * waveguide_length + design_region_width
    Sy = 2 * pml_size + design_region_height + 0.5
    cell_size = Vector3(Sx, Sy)

    pml_layers = [PML(pml_size)]

    fcen = 1 / 1.56
    width = 0.2
    fwidth = width * fcen
    source_center = [-Sx / 2 + pml_size + waveguide_length / 3, 0, 0]
    source_size = Vector3(0, 2, 0)
    kpoint = Vector3(1, 0, 0)
    src = GaussianSource(frequency=fcen, fwidth=fwidth)
    source = [
        EigenModeSource(
            src,
            eig_band=1,
            direction=NO_DIRECTION,
            eig_kpoint=kpoint,
            size=source_size,
            center=source_center,
        )
    ]

    SiO2 = Medium(index=1.44)

    return Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=source,
        symmetries=[Mirror(direction=Y)],
        default_material=SiO2,
        resolution=resolution,
    )


def test_meep_adjoint_optimization() -> None:
    import autograd.numpy as npa
    import numpy as np

    design_region, design_variables, c, Nx, Ny = _example_optim_geometry()

    seed = 240
    np.random.seed(seed)
    x0 = np.random.rand(
        Nx * Ny,
    )

    def J(source, top, bottom):
        power = npa.abs(top / source) ** 2 + npa.abs(bottom / source) ** 2
        return npa.mean(power)

    sim_gf = get_meep_adjoint_optimizer(
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
    ).sim

    meep_sim = _get_sim_from_meep()

    assert sim_gf == meep_sim
