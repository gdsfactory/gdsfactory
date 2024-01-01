from __future__ import annotations

from functools import partial

import flatdict

import gdsfactory as gf
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


@gf.cell_with_child
def add_label_ehva(
    component: gf.Component,
    die: str = "demo",
    prefix_to_type: dict[str, str] = prefix_to_type_default,
    layer: Layer = (66, 0),
    metadata_ignore: list[str] | None = None,
    metadata_include: list[str] | None = None,
) -> gf.Component:
    """Returns new Component with measurement labels.

    Args:
        component: to add labels to.
        die: string.
        port_types: list of port types to label.
        layer: text label layer.
        metadata_ignore: list of settings keys to ignore.
            Works with flatdict setting:subsetting.
        metadata_include: includes settings .
            Works with flatdict setting:subsetting.

    """
    c = gf.Component()
    component = gf.get_component(component)
    ref = c << component
    c.add_ports(ref.ports)
    c.copy_child_info(component)

    metadata_ignore = metadata_ignore or []
    metadata_include = metadata_include or []

    text = f"""DIE NAME:{die}
CIRCUIT NAME:{component.name}
"""
    info = []

    metadata = dict(component.info)
    metadata = flatdict.FlatDict(metadata)
    if metadata:
        info += [
            f"CIRCUITINFO NAME: {k}, VALUE: {v}"
            for k, v in metadata.items()
            if k not in metadata_ignore and isinstance(v, int | float | str)
        ]

    metadata = flatdict.FlatDict(dict(component.settings))
    info += [
        f"CIRCUITINFO NAME: {k}, VALUE: {metadata.get(k)}"
        for k in metadata
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
    c.add(label)
    return c


if __name__ == "__main__":
    add_label_ehva_demo = partial(
        add_label_ehva,
        die="demo_die",
        metadata_include=["grating_coupler:settings:polarization"],
    )

    c = gf.c.straight(length=11)
    c = gf.c.mmi2x2(length_mmi=2.2)
    c = gf.routing.add_fiber_array(
        c,
        get_input_labels_function=None,
        grating_coupler=gf.c.grating_coupler_te,
    )
    c = add_label_ehva_demo(c)

    # add_label_ehva(c, die="demo_die", metadata_include_child=["width_mmi"])
    # add_label_ehva(c, die="demo_die", metadata_include_child=[])

    print(c.labels[0])
    c.show(show_ports=True)
