from __future__ import annotations

from typing import Dict, List, Optional, Union

import meep as mp
import numpy as np

import gdsfactory as gf
from gdsfactory.pdk import get_layer_stack
from gdsfactory.simulation.gmeep.get_material import get_material
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


def get_meep_geometry_from_component(
    component: ComponentSpec,
    layer_stack: Optional[LayerStack] = None,
    material_name_to_meep: Optional[Dict[str, Union[str, float]]] = None,
    wavelength: float = 1.55,
    is_3d: bool = False,
    dispersive: bool = False,
    **kwargs,
) -> List[mp.GeometricObject]:
    """Returns Meep geometry from a gdsfactory component.

    Args:
        component: gdsfactory component.
        layer_stack: for material layers.
        material_name_to_meep: maps layer_stack name to meep material name.
        wavelength: in um.
        is_3d: renders in 3D.
        dispersive: add dispersion.
        kwargs: settings.
    """
    component = gf.get_component(component=component, **kwargs)
    layer_stack = layer_stack or get_layer_stack()

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_material = layer_stack.get_layer_to_material()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    layer_to_sidewall_angle = layer_stack.get_layer_to_sidewall_angle()
    component_with_booleans = layer_stack.get_component_with_derived_layers(component)

    geometry = []
    layer_to_polygons = component_with_booleans.get_polygons(by_spec=True)

    for layer, polygons in layer_to_polygons.items():
        if layer in layer_to_thickness and layer in layer_to_material:
            height = layer_to_thickness[layer] if is_3d else mp.inf
            zmin_um = layer_to_zmin[layer] if is_3d else 0
            # center = mp.Vector3(0, 0, (zmin_um + height) / 2)

            for polygon in polygons:
                vertices = [mp.Vector3(p[0], p[1], zmin_um) for p in polygon]
                material_name = layer_to_material[layer]

                if material_name:
                    material = get_material(
                        name=material_name,
                        dispersive=dispersive,
                        material_name_to_meep=material_name_to_meep,
                        wavelength=wavelength,
                    )
                    geometry.append(
                        mp.Prism(
                            vertices=vertices,
                            height=height,
                            sidewall_angle=layer_to_sidewall_angle[layer]
                            if is_3d
                            else 0,
                            material=material,
                            # center=center
                        )
                    )
    return geometry


def get_meep_geometry_from_cross_section(
    cross_section: CrossSectionSpec,
    extension_length: Optional[float] = None,
    layer_stack: Optional[LayerStack] = None,
    material_name_to_meep: Optional[Dict[str, Union[str, float]]] = None,
    wavelength: float = 1.55,
    dispersive: bool = False,
    **kwargs,
) -> List[mp.GeometricObject]:
    x = gf.get_cross_section(cross_section=cross_section, **kwargs)

    x_sections = [
        gf.Section(offset=x.offset, layer=x.layer, width=x.width),
        *x.sections,
    ]

    layer_stack = layer_stack or get_layer_stack()

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_material = layer_stack.get_layer_to_material()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    layer_to_sidewall_angle = layer_stack.get_layer_to_sidewall_angle()

    geometry = []
    for section in x_sections:
        print(f"section: {section}")
        layer = gf.get_layer(section.layer)

        if layer in layer_to_thickness and layer in layer_to_material:
            height = layer_to_thickness[layer]
            width = section.width
            offset = section.offset

            zmin_um = layer_to_zmin[layer] + (0 if height > 0 else -height)
            # center = mp.Vector3(0, 0, (zmin_um + height) / 2)

            material_name = layer_to_material[layer]
            material = get_material(
                name=material_name,
                dispersive=dispersive,
                material_name_to_meep=material_name_to_meep,
                wavelength=wavelength,
            )
            index = material.epsilon(1 / wavelength)[0, 0] ** 0.5
            print(f"add {material_name!r} layer with index {index}")
            # Don't need to use prism unless using sidewall angles
            if layer in layer_to_sidewall_angle:
                # If using a prism, all dimensions need to be finite
                xspan = extension_length or 1
                p = mp.Prism(
                    vertices=[
                        mp.Vector3(x=-xspan / 2, y=-width / 2, z=zmin_um),
                        mp.Vector3(x=-xspan / 2, y=width / 2, z=zmin_um),
                        mp.Vector3(x=xspan / 2, y=width / 2, z=zmin_um),
                        mp.Vector3(x=xspan / 2, y=-width / 2, z=zmin_um),
                    ],
                    height=height,
                    center=mp.Vector3(y=offset, z=height / 2 + zmin_um),
                    sidewall_angle=np.deg2rad(layer_to_sidewall_angle[layer]),
                    material=material,
                )
                geometry.append(p)

            else:
                xspan = extension_length or mp.inf
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(xspan, width, height),
                        material=material,
                        center=mp.Vector3(y=offset, z=height / 2 + zmin_um),
                    )
                )
    return geometry


if __name__ == "__main__":
    import gdsfactory.simulation.gmeep as gm
    import matplotlib.pyplot as plt

    c = gf.components.taper_strip_to_ridge_trenches()
    sp = gm.write_sparameters_meep(
        c, run=False, ymargin_top=3, ymargin_bot=3, is_3d=True
    )
    plt.show()
    # c.show(show_ports=True)
