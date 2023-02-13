"""Add label YAML."""
from __future__ import annotations

from typing import List, Optional

import flatdict
import pydantic

import gdsfactory as gf
from gdsfactory.name import clean_name
from gdsfactory.typings import LayerSpec

ignore = [
    "cross_section",
    "decorator",
    "cross_section1",
    "cross_section2",
    "contact",
    "pad",
]
port_prefixes = [
    "opt_",
    "elec_",
]


@pydantic.validate_arguments
def add_label_yaml(
    component: gf.Component,
    port_prefixes: List[str] = ("opt_", "_elec"),
    layer: LayerSpec = "LABEL",
    metadata_ignore: Optional[List[str]] = ignore,
    metadata_include_parent: Optional[List[str]] = None,
    metadata_include_child: Optional[List[str]] = None,
) -> gf.Component:
    """Returns Component with measurement label.

    Args:
        component: to add labels to.
        port_types: list of port types to label.
        layer: text label layer.
        metadata_ignore: list of settings keys to ignore.
            Works with flatdict setting:subsetting.
        metadata_include_parent: parent metadata keys to include.
            Works with flatdict setting:subsetting.
        metadata_include_child: child metadata keys to include.

    """
    from gdsfactory.pdk import get_layer

    metadata_ignore = metadata_ignore or []
    metadata_include_parent = metadata_include_parent or []
    metadata_include_child = metadata_include_child or []

    text = f"""component_name: {component.name}
polarization: {component.metadata.get('polarization')}
wavelength: {component.metadata.get('wavelength')}
settings:
"""
    info = []
    layer = get_layer(layer)

    # metadata = component.metadata_child.changed
    metadata = component.metadata_child.get("changed")
    if metadata:
        info += [
            f"  {k}: {v}"
            for k, v in metadata.items()
            if k not in metadata_ignore and isinstance(v, (int, float, str))
        ]

    metadata = (
        flatdict.FlatDict(component.metadata.get("full"))
        if component.metadata.get("full")
        else {}
    )
    info += [
        f"  {clean_name(k)}: {metadata.get(k)}"
        for k in metadata_include_parent
        if metadata.get(k)
    ]

    metadata = (
        flatdict.FlatDict(component.metadata_child.get("full"))
        if component.metadata_child.get("full")
        else {}
    )
    info += [
        f"  {clean_name(k)}: {metadata.get(k)}"
        for k in metadata_include_child
        if metadata.get(k)
    ]

    info += ["ports:\n"]

    ports_info = []
    if component.ports:
        for port_prefix in port_prefixes:
            for port in component.get_ports_list(prefix=port_prefix):
                ports_info += []
                ports_info += [f"  {port.name}:"]
                s = f"    {port.to_yaml()}"
                s = s.split("\n")
                ports_info += ["    \n    ".join(s)]

    text += "\n".join(info)
    text += "\n".join(ports_info)

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
    from omegaconf import OmegaConf

    c = gf.c.straight(length=11)
    c = gf.c.mmi2x2(length_mmi=2.2)
    c = gf.routing.add_fiber_array(
        c,
        get_input_labels_function=None,
        grating_coupler=gf.components.grating_coupler_te,
        decorator=add_label_yaml,
    )
    print(c.labels[0].text)
    d = OmegaConf.create(c.labels[0].text)
    c.show(show_ports=True)
