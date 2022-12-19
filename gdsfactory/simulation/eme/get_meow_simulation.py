from itertools import permutations

import meow as mw
import numpy as np

from gdsfactory.simulation.gmsh.parse_layerstack import list_unique_layerstack_z
from gdsfactory.tech import LAYER, LayerStack
from gdsfactory.types import ComponentSpec

"""Conversion between gdsfactory material names and meow materials class."""
gdsfactory_to_meow_materials = {
    "si": mw.silicon,
    "sio2": mw.silicon_oxide,
}


def add_global_layers(component, layerstack):
    """Adds bbox polygons for global layers."""
    bbox = component.bbox
    for layername, layer in layerstack.layers.items():
        if layer.layer == LAYER.WAFER:
            component.add_ref(
                gf.components.box(bbox[0, 0], bbox[0, 1], bbox[1, 0], bbox[1, 1])
            )
        else:
            continue
    return component


def layerstack_to_extrusion(layerstack: LayerStack):
    """Convert LayerStack to meow extrusions."""
    extrusions = {}
    for layername, layer in layerstack.layers.items():
        if layer.layer not in extrusions.keys():
            extrusions[layer.layer] = []
        extrusions[layer.layer].append(
            mw.GdsExtrusionRule(
                material=gdsfactory_to_meow_materials[layer.material],
                h_min=layer.zmin,
                h_max=layer.zmin + layer.thickness,
                mesh_order=layer.info["mesh_order"],
            )
        )
    return extrusions


def get_eme_cells(
    component,
    extrusion_rules,
    num_cells: int = 10,
    xspan: float = 2.0,
    xoffset: float = 0,
    xres: int = 100,
    yspan: float = 2.0,
    yoffset: float = 0,
    yres: int = 100,
):
    """Get meow cells from extruded component.

    Arguments:
        structs: meow structs
        extrusion rules: from layerstack_to_extrusion
        num_cells: number of slices
        xspan: size of the horizontal cross-sectional simulation region
        xoffset: center of the horizontal cross-sectional simulation region
        xres: number of mesh points for the horizontal cross-sectional simulation region
        yspan: size of the vertical cross-sectional simulation region
        yoffset: center of the vertical cross-sectional simulation region
        yres: number of mesh points for the vertical cross-sectional simulation region
    """
    structs = mw.extrude_gds(component, extrusion_rules)
    bbox = component.bbox
    Ls = [np.diff(bbox[:, 0]).item() / num_cells for _ in range(num_cells)]
    return mw.create_cells(
        structures=structs,
        mesh=mw.Mesh2d(
            x=np.linspace(xoffset - xspan / 2, xoffset + xspan / 2, xres),
            y=np.linspace(yoffset - yspan / 2, yoffset + yspan / 2, yres),
        ),
        Ls=Ls,
    )


def eme_calculation(
    component: ComponentSpec,
    layerstack: LayerStack,
    num_modes: int = 4,
    wavelength: float = 1.55,
    temperature: float = 25.0,
    num_cells: int = 10,
    xspan: float = None,
    xoffset: float = 0.0,
    xres: int = 100,
    yspan: float = None,
    yoffset: float = 0.0,
    yres: int = 100,
    validate_component: bool = True,
):
    """Computes multimode 2-port S-parameters for a gdsfactory component, assuming port 1 is at the left boundary and port 2 at the right boundary.

    Arguments:
        component: gdsfactory component
        layerstack: gdsfactory layerstack
        num_modes: number of modes to compute for the eigenmode expansion
        wavelength: wavelength in microns (for FDE, and for material properties)
        temperature: temperature in C (for material properties)
        num_cells: number of component slices along the propagation direction for the EME
        xspan: size of the horizontal cross-sectional simulation region
        xoffset: center of the horizontal cross-sectional simulation region
        xres: number of mesh points for the horizontal cross-sectional simulation region
        yspan: size of the vertical cross-sectional simulation region
        yoffset: center of the vertical cross-sectional simulation region
        yres: number of mesh points for the vertical cross-sectional simulation region
        validate_component: whether raise errors if the component ports are of the wrong number and orientation

    Returns:
        S-parameters in form o1@0:o2@0, etc.
    """
    # Check component validity
    if validate_component:
        optical_ports = [
            port
            for portname, port in component.ports.items()
            if port.port_type == "optical"
        ]
        if len(optical_ports) != 2:
            raise ValueError(
                "Component provided to MEOW does not have exactly 2 optical ports."
            )
        elif component.ports["o1"].orientation != 180:
            raise ValueError("Component port o1 does not face westward (180 deg).")
        elif component.ports["o2"].orientation != 0:
            raise ValueError("Component port o2 does not face eastward (0 deg).")

    # Preprocess component and layerstack
    component = add_global_layers(component, layerstack)
    extrusion_rules = layerstack_to_extrusion(layerstack)
    xspan = xspan or component.bbox[0, 1] + 2
    zs = list_unique_layerstack_z(layerstack)
    yspan = yspan or np.max(zs) - np.min(zs)

    # Get EME cells
    cells = get_eme_cells(
        component,
        extrusion_rules,
        num_cells=num_cells,
        xspan=xspan,
        xoffset=xoffset,
        xres=xres,
        yspan=yspan,
        yoffset=yoffset,
        yres=yres,
    )

    # Get EME cross-sections
    env = mw.Environment(wl=wavelength, T=temperature)
    css = [mw.CrossSection(cell=cell, env=env) for cell in cells]

    # Compute modes
    modes = [mw.compute_modes(cs, num_modes=num_modes) for cs in css]

    # Compute EME
    S, port_map = mw.compute_s_matrix(modes)

    # Convert coefficients to existing format
    meow_to_gf_keys = {
        "left": "o1",
        "right": "o2",
    }
    sp = {}
    for port1, port2 in permutations(port_map.values(), 2):
        value = S[port1, port2]
        meow_key1 = [k for k, v in port_map.items() if v == port1][0]
        meow_port1, meow_mode1 = meow_key1.split("@")
        meow_key2 = [k for k, v in port_map.items() if v == port2][0]
        meow_port2, meow_mode2 = meow_key2.split("@")
        sp[
            f"{meow_to_gf_keys[meow_port1]}@{meow_mode1},{meow_to_gf_keys[meow_port2]}@{meow_mode2}"
        ] = value
    sp["wavelengths"] = wavelength

    return sp


if __name__ == "__main__":

    import gdsfactory as gf

    c = gf.components.taper_cross_section_linear()
    c.show()

    from gdsfactory.tech import get_layer_stack_generic

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
                "box",
                "clad",
            )
        }
    )

    sp = eme_calculation(component=c, layerstack=filtered_layerstack)

    import pprint

    pprint.pprint(sp)
