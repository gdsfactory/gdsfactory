import time
from typing import Tuple

import numpy as np
from femwell.maxwell.waveguide import compute_modes, Modes
from skfem import (
    Basis,
    ElementTriN2,
    ElementTriP0,
    ElementTriP2,
    Mesh,
)

import gdsfactory as gf
from gdsfactory.pdk import get_layer_stack, get_material_index
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, PathType

mesh_filename = "mesh.msh"


def load_mesh_basis(mesh_filename: PathType):
    mesh = Mesh.load(mesh_filename)
    basis = Basis(mesh, ElementTriN2() * ElementTriP2())
    basis0 = basis.with_element(ElementTriP0())
    return mesh, basis0


def compute_cross_section_modes(
    cross_section: CrossSectionSpec,
    layerstack: LayerStack,
    wavelength: float = 1.55,
    num_modes: int = 4,
    order: int = 1,
    radius: float = np.inf,
    wafer_padding: float = 2.0,
    **kwargs,
) -> Modes:
    """Calculate effective index of a cross-section.

    Defines a "straight" component of the cross_section, and calls compute_component_slice_modes.

    Args:
        cross_section:
        layerstack:
        wavelength: in um.
        num_modes: to compute.
        order: order of the mesh elements. 1: linear, 2: quadratic.
        radius: defaults to inf.
        wafer_padding: in um.

    Keyword Args:
        solver: can be slepc or scipy.
        resolutions (Dict): Pairs {"layername": {"resolution": float, "distance": "float}}
            to roughly control mesh refinement within and away from entity, respectively.
        mesh_scaling_factor (float): factor multiply mesh geometry by.
        default_resolution_min (float): gmsh minimal edge length.
        default_resolution_max (float): gmsh maximal edge length.
        background_tag (str): name of the background layer to add (default: no background added).
        background_padding (Tuple): [xleft, ydown, xright, yup] distances to add to the components and to fill with background_tag.
        global_meshsize_array: np array [x,y,z,lc] to parametrize the mesh.
        global_meshsize_interpolant_func: interpolating function for global_meshsize_array.
        extra_shapes_dict: Optional[OrderedDict] of {key: geo} with key a label and geo a shapely (Multi)Polygon or (Multi)LineString of extra shapes to override component.
        merge_by_material: boolean, if True will merge polygons from layers with the same layer.material. Physical keys will be material in this case.

    """
    # Get meshable component from cross-section
    c = gf.components.straight(length=10, cross_section=cross_section)
    bounds = c.bbox
    dx = np.diff(bounds[:, 0])[0]

    xsection_bounds = [
        [dx / 2, bounds[0, 1] - wafer_padding],
        [dx / 2, bounds[1, 1] + wafer_padding],
    ]

    # Mesh as component
    return compute_component_slice_modes(
        component=c,
        xsection_bounds=xsection_bounds,
        layerstack=layerstack,
        wavelength=wavelength,
        num_modes=num_modes,
        order=order,
        radius=radius,
        wafer_padding=wafer_padding,
        **kwargs,
    )


def compute_component_slice_modes(
    component: ComponentSpec,
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    layerstack: LayerStack,
    wavelength: float = 1.55,
    num_modes: int = 4,
    order: int = 1,
    radius: float = np.inf,
    wafer_padding: float = 2.0,
    solver: str = "scipy",
    **kwargs,
) -> Modes:
    """Calculate effective index of component slice.

    Args:
        component: gdsfactory component.
        xsection_bounds: xy line defining where to take component cross_section.
        layerstack: gdsfactory layerstack.
        wavelength: wavelength (um).
        num_modes: number of modes to return.
        order: order of the mesh elements. 1: linear, 2: quadratic.
        radius: bend radius of the cross-section.
        wafer_padding: padding beyond bbox to add to WAFER layers.
        bend_radius: bend radius for mode solving (typically called from cross_section)
        solver: can be slepc or scipy.

    Keyword Args:
        resolutions (Dict): Pairs {"layername": {"resolution": float, "distance": "float}}
            to roughly control mesh refinement within and away from entity, respectively.
        mesh_scaling_factor (float): factor multiply mesh geometry by.
        default_resolution_min (float): gmsh minimal edge length.
        default_resolution_max (float): gmsh maximal edge length.
        background_tag (str): name of the background layer to add (default: no background added).
        background_padding (Tuple): [xleft, ydown, xright, yup] distances to add to the components and to fill with background_tag.
        global_meshsize_array: np array [x,y,z,lc] to parametrize the mesh.
        global_meshsize_interpolant_func: interpolating function for global_meshsize_array.
        extra_shapes_dict: Optional[OrderedDict] of {key: geo} with key a label and geo a shapely (Multi)Polygon or (Multi)LineString of extra shapes to override component.
        merge_by_material: boolean, if True will merge polygons from layers with the same layer.material. Physical keys will be material in this case.
    """

    # Mesh
    mesh = component.to_gmsh(
        type="uz",
        xsection_bounds=xsection_bounds,
        layer_stack=layerstack,
        filename=mesh_filename,
        wafer_padding=wafer_padding,
        **kwargs,
    )

    # Assign materials to mesh elements
    mesh, basis0 = load_mesh_basis(mesh_filename)
    epsilon = basis0.zeros(dtype=complex)
    for layername, layer in layerstack.layers.items():
        if layername in mesh.subdomains.keys():
            epsilon[basis0.get_dofs(elements=layername)] = (
                get_material_index(layer.material, wavelength) ** 2
            )
        if "background_tag" in kwargs:
            epsilon[basis0.get_dofs(elements=kwargs["background_tag"])] = (
                get_material_index(kwargs["background_tag"], wavelength) ** 2
            )

    # Mode solve
    modes = compute_modes(
        basis0,
        epsilon,
        wavelength=wavelength,
        mu_r=1,
        num_modes=num_modes,
        order=order,
        radius=radius,
        solver=solver,
    )
    return modes


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    PDK = gf.get_generic_pdk()
    PDK.activate()

    start = time.time()

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                "core",
                "clad",
                "slab90",
                "box",
            )
        }
    )

    filtered_layerstack.layers["core"].thickness = 0.2

    resolutions = {
        "core": {"resolution": 0.02, "distance": 2},
        "clad": {"resolution": 0.2, "distance": 1},
        "box": {"resolution": 0.2, "distance": 1},
        "slab90": {"resolution": 0.05, "distance": 1},
    }

    cross_section = False
    cross_section = True

    if cross_section:
        modes = compute_cross_section_modes(
            cross_section="rib",
            layerstack=filtered_layerstack,
            wavelength=1.55,
            num_modes=4,
            order=1,
            radius=np.inf,
            resolutions=resolutions,
        )
        mode = modes[0]
        mode.show(mode.E.real, colorbar=True, direction="x")
    else:
        component = gf.components.coupler_full(dw=0)
        component.show()

        modes = compute_component_slice_modes(
            component,
            [[0, -3], [0, 3]],
            layerstack=filtered_layerstack,
            wavelength=1.55,
            num_modes=4,
            order=1,
            radius=np.inf,
            mesh_filename="./mesh.msh",
            resolutions=resolutions,
        )

        print(modes)
        print(modes[0].te_fraction)

        modes[0].show(np.real(modes[0].E))
        modes[0].show(np.imag(modes[0].E))

        modes[0].show(np.real(modes[0].H))
        modes[0].show(np.imag(modes[0].H))

        integrals = np.zeros((len(modes),) * 2, dtype=complex)

        for i in range(len(modes)):
            for j in range(len(modes)):
                integrals[i, j] = modes[i].calculate_overlap(modes[j])

        plt.imshow(np.real(integrals))
        plt.colorbar()
        plt.show()

        # Create basis to select a certain simulation extent
        def sel_fun(x):
            return (x[0] < 0) * (x[0] > -1) * (x[1] > 0) * (x[1] < 0.5)

        print(modes.sorted(lambda mode: mode.calculate_power(elements=sel_fun)))
