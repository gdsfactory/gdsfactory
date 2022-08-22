import copy as python_copy

import gdspy
from phidl.device_layout import CellArray

from gdsfactory.component import Component, ComponentReference


def copy(
    D: Component,
) -> Component:
    """Returns a deep copy of a Component.

    Args:
        D: component.
    """
    D_copy = Component()
    D_copy.info = python_copy.deepcopy(D.info)
    for ref in D.references:
        if isinstance(ref, gdspy.CellReference):
            new_ref = ComponentReference(
                ref.parent,
                origin=ref.origin,
                rotation=ref.rotation,
                magnification=ref.magnification,
                x_reflection=ref.x_reflection,
            )
            new_ref.owner = D_copy
            new_ref.name = ref.name
        elif isinstance(ref, gdspy.CellArray):
            new_ref = CellArray(
                device=ref.parent,
                columns=ref.columns,
                rows=ref.rows,
                spacing=ref.spacing,
                origin=ref.origin,
                rotation=ref.rotation,
                magnification=ref.magnification,
                x_reflection=ref.x_reflection,
            )
            new_ref.name = ref.name
        else:
            raise ValueError(f"Got a reference of non-standard type: {type(ref)}")
        D_copy.add(new_ref)

    for port in D.ports.values():
        D_copy.add_port(port=port)
    for poly in D.polygons:
        D_copy.add_polygon(poly)
    for path in D.paths:
        D_copy.add(path)
    for label in D.labels:
        D_copy.add_label(
            text=label.text,
            position=label.position,
            layer=(label.layer, label.texttype),
        )
    return D_copy
