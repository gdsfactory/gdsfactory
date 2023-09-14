from __future__ import annotations

from functools import partial

import flatdict
import pydantic

import gdsfactory as gf
from gdsfactory.name import clean_name
from gdsfactory.snap import snap_to_grid as snap
from gdsfactory.typings import Layer

ignore = (
    "cross_section",
    "decorator",
    "cross_section1",
    "cross_section2",
    "contact",
    "pad",
)
prefix_to_type_default = {
    "opt": "OPTICALPORT",
    "pad": "ELECTRICALPORT",
    "vertical_dc": "ELECTRICALPORT",
    "optical": "OPTICALPORT",
    "loopback": "OPTICALPORT",
}


@pydantic.validate_call
def add_label_ehva(
    component: gf.Component,
    die: str = "demo",
    prefix_to_type: dict[str, str] = prefix_to_type_default,
    layer: Layer = (66, 0),
    metadata_ignore: list[str] | None = None,
    metadata_include_parent: list[str] | None = None,
    metadata_include_child: list[str] | None = None,
) -> gf.Component:
    """Returns Component with measurement labels.

    Args:
        component: to add labels to.
        die: string.
        port_types: list of port types to label.
        layer: text label layer.
        metadata_ignore: list of settings keys to ignore.
            Works with flatdict setting:subsetting.
        metadata_include_parent: includes parent metadata.
            Works with flatdict setting:subsetting.

    """
    metadata_ignore = metadata_ignore or []
    metadata_include_parent = metadata_include_parent or []
    metadata_include_child = metadata_include_child or []

    text = f"""DIE NAME:{die}
CIRCUIT NAME:{component.name}
"""
    info = []

    metadata = component.metadata_child["changed"]
    if metadata:
        info += [
            f"CIRCUITINFO NAME: {k}, VALUE: {v}"
            for k, v in metadata.items()
            if k not in metadata_ignore and isinstance(v, int | float | str)
        ]

    metadata = flatdict.FlatDict(component.metadata["full"])
    info += [
        f"CIRCUITINFO NAME: {clean_name(k)}, VALUE: {metadata.get(k)}"
        for k in metadata_include_parent
        if metadata.get(k)
    ]

    metadata = flatdict.FlatDict(component.metadata_child["full"])
    info += [
        f"CIRCUITINFO NAME: {k}, VALUE: {metadata.get(k)}"
        for k in metadata_include_child
        if metadata.get(k)
    ]

    text += "\n".join(info)
    text += "\n"

    info = []
    if component.ports:
        for prefix, port_type_ehva in prefix_to_type.items():
            info += [
                f"{port_type_ehva} NAME: {port.name} TYPE: {port_type_ehva}, "
                f"POSITION RELATIVE:({snap(port.x)}, {snap(port.y)}),"
                f" ORIENTATION: {port.orientation}"
                for port in component.get_ports_list(prefix=prefix)
            ]
    text += "\n".join(info)

    label = gf.Label(
        text=text,
        origin=(0, 0),
        anchor="o",
        layer=layer[0],
        texttype=layer[1],
    )
    component.add(label)
    return component


if __name__ == "__main__":
    add_label_ehva_demo = partial(
        add_label_ehva,
        die="demo_die",
        metadata_include_parent=["grating_coupler:settings:polarization"],
    )

    c = gf.c.straight(length=11)
    c = gf.c.mmi2x2(length_mmi=2.2)
    c = gf.routing.add_fiber_array(
        c,
        get_input_labels_function=None,
        grating_coupler=gf.c.grating_coupler_te,
        decorator=add_label_ehva_demo,
    )

    # add_label_ehva(c, die="demo_die", metadata_include_child=["width_mmi"])
    # add_label_ehva(c, die="demo_die", metadata_include_child=[])

    print(c.labels[0])
    c.show(show_ports=True)
