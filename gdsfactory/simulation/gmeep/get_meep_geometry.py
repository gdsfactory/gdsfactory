from typing import Dict, Optional, Union

import meep as mp

import gdsfactory as gf
from gdsfactory.simulation.gmeep.get_material import get_material
from gdsfactory.types import ComponentSpec, LayerStack


def get_meep_geometry_from_component(
    component: ComponentSpec,
    layer_stack: Optional[LayerStack] = None,
    material_name_to_meep: Optional[Dict[str, Union[str, float]]] = None,
    wavelength: float = 1.55,
    is_3d: bool = False,
    dispersive: bool = False,
    **kwargs
):
    component = gf.get_component(component=component, **kwargs)
    component_ref = component.ref()

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_material = layer_stack.get_layer_to_material()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    layer_to_sidewall_angle = layer_stack.get_layer_to_sidewall_angle()

    geometry = []
    layer_to_polygons = component_ref.get_polygons(by_spec=True)
    for layer, polygons in layer_to_polygons.items():
        if layer in layer_to_thickness and layer in layer_to_material:
            height = layer_to_thickness[layer] if is_3d else mp.inf
            zmin_um = layer_to_zmin[layer] if is_3d else 0
            # center = mp.Vector3(0, 0, (zmin_um + height) / 2)

            for polygon in polygons:
                vertices = [mp.Vector3(p[0], p[1], zmin_um) for p in polygon]
                material_name = layer_to_material[layer]
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
                        sidewall_angle=layer_to_sidewall_angle[layer],
                        material=material,
                        # center=center
                    )
                )
    return geometry
