from typing import Dict

import numpy as np
import skfem
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP0,
    ElementTriP1,
    LinearForm,
    asm,
    condense,
    solve,
)
from skfem.helpers import dot, grad


def solve_thermal(
    mesh_filename: str,
    thermal_conductivity: Dict[str, float],
    specific_conductivity: Dict[str, float],
    currents: Dict[str, float],
):
    """Thermal simulation.

    Args:
        mesh_filename: Name of the mesh to load
        thermal_conductivity: thermal conductivity in W/m‧K
        specific_conductivity: specific conductivity in S/m
        currents: current flowing through the layer in A

    Returns:
        nothing yet, WIP
    """
    mesh = skfem.Mesh.load(mesh_filename)

    @BilinearForm
    def conduction(u, v, w):
        return dot(w["thermal_conductivity"] * u.grad, grad(v))

    @LinearForm
    def unit_load(v, _):
        return v

    basis = Basis(mesh, ElementTriP1())
    joule_heating_rhs = basis.zeros()
    for domain, current in currents.items():  # sum up the sources for the heating
        core_basis = Basis(mesh, basis.elem, elements=mesh.subdomains[domain])
        asm_core_unit_load = asm(unit_load, core_basis)
        core_area = np.sum(asm_core_unit_load)
        joule_heating = (current / core_area) ** 2 / specific_conductivity[domain]
        joule_heating_rhs += joule_heating * asm_core_unit_load

    basis0 = basis.with_element(ElementTriP0())
    thermal_conductivity_p0 = basis0.zeros()
    for domain in thermal_conductivity.keys():
        thermal_conductivity_p0[
            basis0.get_dofs(elements=domain)
        ] = thermal_conductivity[domain]
    thermal_conductivity_p0 *= 1e-12  # 1e-12 -> conversion from 1/m^2 -> 1/um^2

    thermal_conductivity_lhs = asm(
        conduction,
        basis,
        thermal_conductivity=basis0.interpolate(thermal_conductivity_p0),
    )

    temperature = solve(
        *condense(
            thermal_conductivity_lhs,
            joule_heating_rhs,
            D=basis.get_dofs(mesh.boundaries["bottom"]),
        )
    )

    from skfem.visuals.matplotlib import draw, plot

    ax = draw(mesh)
    ax.show()

    ax = draw(mesh, boundaries_only=True)
    plot(basis0, thermal_conductivity_p0 * 1e12, ax=ax, colorbar=True)
    ax.figure.set_size_inches(10, 7)
    ax.set_axis_on()
    ax.figure.tight_layout()
    ax.show()

    ax = draw(mesh, boundaries_only=True)
    plot(basis, temperature, ax=ax, colorbar=True, shading="gouraud")
    ax.figure.set_size_inches(10, 7)
    ax.set_axis_on()
    ax.figure.tight_layout()
    ax.show()


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.simulation.gmsh.mesh2D import mesh2D

    gf.tech.LAYER_STACK.layers["heater"].thickness = 0.13
    gf.tech.LAYER_STACK.layers["heater"].zmin = 2.2

    heater1 = gf.components.straight_heater_metal(length=50, heater_width=2)
    heater2 = gf.components.straight_heater_metal(length=50, heater_width=2).move(
        [0, -10]
    )

    heaters = gf.Component("heaters")
    heaters << heater1
    heaters << heater2
    heaters.show()

    geometry = mesh2D(
        heaters,
        ((25, -25), (25, 25)),
        base_resolution=0.4,
        exclude_layers=((1, 10),),
        padding=(10, 10, 1, 1),
        refine_resolution={(1, 0): 0.05, (47, 0): 0.02},
    )

    import gmsh

    gmsh.write("mesh.msh")
    gmsh.clear()
    geometry.__exit__()

    solve_thermal(
        mesh_filename="mesh.msh",
        thermal_conductivity={"(47, 0)": 28, "oxide": 1.38, "(1, 0)": 148},
        specific_conductivity={"(47, 0)_0": 2.3e6},
        currents={"(47, 0)_0": 0.007},
    )
