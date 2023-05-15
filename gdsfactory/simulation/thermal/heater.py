from __future__ import annotations

from typing import Dict, Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skfem
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse.linalg import splu
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP0,
    ElementTriP1,
    LinearForm,
    asm,
    condense,
    penalize,
    solve,
)
from skfem.helpers import dot, grad
from skfem.visuals.matplotlib import draw, plot


def solve_thermal(
    mesh_filename: str,
    thermal_conductivity: Dict[str, float],
    specific_conductivity: Dict[str, float],
    thermal_diffusivity: Dict[str, float],
    currents: Dict[str, float],
) -> None:
    """Thermal simulation.

    Args:
        mesh_filename: Name of the mesh to load.
        thermal_conductivity: thermal conductivity in W/m‧K.
        specific_conductivity: specific conductivity in S/m.
        thermal_diffusivity:
        currents: current flowing through the layer in A.

    Returns:
        nothing yet, WIP.
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
    for domain in thermal_conductivity:
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

    ax = draw(mesh)
    ax.show()

    ax = draw(mesh, boundaries_only=True)
    plot(basis0, thermal_conductivity_p0 * 1e12, ax=ax, colorbar=True)
    ax.figure.set_size_inches(10, 7)
    ax.set_axis_on()
    ax.figure.tight_layout()
    ax.show()

    ax = draw(mesh, boundaries_only=True)
    plot(basis, joule_heating_rhs, ax=ax, colorbar=True, shading="gouraud")
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

    print("max temp steady", np.max(temperature))
    print("average team steady", np.mean(temperature))

    thermal_diffusivity_p0 = basis0.zeros()
    for domain in thermal_diffusivity:
        thermal_diffusivity_p0[basis0.get_dofs(elements=domain)] = thermal_diffusivity[
            domain
        ]

    thermal_diffusivity_p0 *= 1e12  # 1e-12 -> conversion from m^2 -> um^2

    @BilinearForm
    def diffusivity_laplace(u, v, w):
        return dot(grad(u) * w["thermal_conductivity"], grad(v))

    @BilinearForm
    def mass(u, v, w):
        return u * v / (w["thermal_diffusivity"] / w["thermal_conductivity"])

    L = asm(
        diffusivity_laplace,
        basis,
        thermal_diffusivity=basis0.interpolate(thermal_diffusivity_p0),
        thermal_conductivity=basis0.interpolate(thermal_conductivity_p0),
    )
    M = asm(
        mass,
        basis,
        thermal_diffusivity=basis0.interpolate(thermal_diffusivity_p0),
        thermal_conductivity=basis0.interpolate(thermal_conductivity_p0),
    )

    dt = 0.1e-6
    theta = 0.5  # Crank–Nicolson
    steps = 200

    L0, M0 = penalize(L, M, D=basis.get_dofs(mesh.boundaries["bottom"]))
    A = M0 + theta * L0 * dt
    B = M0 - (1 - theta) * L0 * dt

    backsolve = splu(A.T).solve  # .T as splu prefers CSC

    def evolve(
        t: float, u: np.ndarray, heating: np.ndarray
    ) -> Iterator[Tuple[float, np.ndarray]]:
        i = 0
        while True:
            t_temperature[i] = t, np.mean(u)
            i += 1
            yield t, u
            t, u = t + dt, backsolve(B @ u + heating * dt)

    ax = draw(mesh, boundaries_only=True)
    ax.set_axis_on()
    ax = plot(mesh, temperature, ax=ax, shading="gouraud")
    title = ax.set_title("t = 0.00")
    field = ax.get_children()[1]  # vertex-based temperature-colour
    fig = ax.get_figure()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(field, cax=cax)

    def update(event):
        t, u = event
        title.set_text(f"$t$ = {t * 1e6:.2f}us")
        field.set_array(u)

    t_temperature = np.zeros((steps + 2, 2))
    animation = FuncAnimation(
        fig,
        update,
        evolve(0.0, temperature * 0.01, joule_heating_rhs),
        repeat=False,
        interval=30,
        save_count=steps,
    )
    animation.save("heater_up.gif", "imagemagick")
    t_temperature_up = t_temperature

    t_temperature = np.zeros((steps + 2, 2))
    animation = FuncAnimation(
        fig,
        update,
        evolve(0.0, temperature, 0),
        repeat=False,
        interval=30,
        save_count=steps,
    )
    animation.save("heater_down.gif", "imagemagick")
    t_temperature_down = t_temperature

    plt.figure()
    plt.plot(t_temperature[:-1, 0] * 1e6, t_temperature_up[:-1, 1])
    plt.plot(t_temperature[:-1, 0] * 1e6, t_temperature_down[:-1, 1])
    plt.plot(
        t_temperature[:-1, 0] * 1e6, t_temperature[:-1, 1] * 0 + np.mean(temperature)
    )
    plt.xlabel("Time [us]")
    plt.ylabel("Average temperature offset [T]")
    plt.savefig("heater.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import gmsh

    import gdsfactory as gf
    from gdsfactory.simulation.gmsh.mesh2D import mesh2D

    gf.generic_tech.LAYER_STACK.layers["heater"].thickness = 0.13
    gf.generic_tech.LAYER_STACK.layers["heater"].zmin = 2.2
    print(gf.generic_tech.LAYER_STACK.layers.keys())
    # gf.generic_tech.LAYER_STACK.layers["core"].thickness = 2

    heater1 = gf.components.straight_heater_metal(length=50, heater_width=2)
    heater2 = gf.components.straight_heater_metal(length=50, heater_width=2).move(
        [0, -10]
    )

    heaters = gf.Component("heaters")
    heaters << heater1
    # heaters << heater2
    heaters.show()

    geometry = mesh2D(
        heaters,
        ((25, -25), (25, 25)),
        base_resolution=0.4,
        exclude_layers=((1, 10),),
        padding=(10, 10, 1, 1),
        refine_resolution={(1, 0): 0.01, (47, 0): 0.005},
    )

    gmsh.write("mesh.msh")
    gmsh.clear()
    geometry.__exit__()

    solve_thermal(
        mesh_filename="mesh.msh",
        thermal_conductivity={"(47, 0)": 28, "oxide": 1.38, "(1, 0)": 148},
        specific_conductivity={"(47, 0)_0": 2.3e6},
        thermal_diffusivity={
            "(47, 0)": 28 / 598 / 5240,
            "oxide": 1.38 / 709 / 2203,
            "(1, 0)": 148 / 711 / 2330,
        },
        # specific_heat={"(47, 0)_0": 598, 'oxide': 709, '(1, 0)': 711},
        # density={"(47, 0)_0": 5240, 'oxide': 2203, '(1, 0)': 2330},
        currents={"(47, 0)_0": 0.007},
    )
