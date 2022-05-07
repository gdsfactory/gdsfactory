from typing import List, Optional, Tuple

import flatdict
import pydantic

import gdsfactory as gf
from gdsfactory.name import clean_name
from gdsfactory.snap import snap_to_grid as snap
from gdsfactory.types import Layer, Strs


class Dft(pydantic.BaseModel):
    pad_size: Tuple[int, int] = (100, 100)
    pad_pitch: int = 125
    pad_width: int = 100
    pad_gc_spacing_opposed: int = 500
    pad_gc_spacing_adjacent: int = 1000


DFT = Dft()


port_types_all = ("vertical_te", "dc", "optical", "loopback")
ignore = ("cross_section", "decorator", "cross_section1", "cross_section2")


@pydantic.validate_arguments
def add_label_ehva(
    component: gf.Component,
    die: str = "demo",
    port_types: Strs = port_types_all,
    layer: Layer = (66, 0),
    metadata_ignore: Optional[List[str]] = None,
    metadata_include_parent: Optional[List[str]] = None,
    metadata_include_child: Optional[List[str]] = None,
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
COMPONENT NAME:{component.name}
"""
    info = []

    metadata = component.metadata_child.changed
    if metadata:
        info += [
            f"COMPONENTINFO NAME: {k}, VALUE {v}"
            for k, v in metadata.items()
            if k not in metadata_ignore
        ]

    metadata = flatdict.FlatDict(component.metadata.full)
    info += [
        f"COMPONENTINFO NAME: {clean_name(k)}, VALUE {metadata.get(k)}"
        for k in metadata_include_parent
        if metadata.get(k)
    ]

    metadata = flatdict.FlatDict(component.metadata_child.full)
    info += [
        f"COMPONENTINFO NAME: {k}, VALUE {metadata.get(k)}"
        for k in metadata_include_child
        if metadata.get(k)
    ]

    text += "\n".join(info)
    text += "\n"

    info = []
    if component.ports:
        for port_type in port_types:
            info += [
                f"OPTICALPORT TYPE: {port_type}, "
                f"POSITION RELATIVE:({snap(port.x)}, {snap(port.y)}),"
                f" ORIENTATION: {port.orientation}"
                for port in component.get_ports_list(port_type=port_type)
            ]
    text += "\n".join(info)

    component.unlock()
    label = gf.Label(
        text=text,
        position=component.size_info.center,
        anchor="o",
        layer=layer[0],
        texttype=layer[1],
    )
    component.add(label)
    component.lock()
    return component


if __name__ == "__main__":
    c = gf.c.straight(length=11)
    c = gf.c.mmi2x2(length_mmi=2.2)
    c = gf.routing.add_fiber_array(
        c, get_input_labels_function=None, grating_coupler=gf.c.grating_coupler_te
    )

    add_label_ehva(
        c,
        die="demo_die",
        metadata_include_parent=["grating_coupler:settings:polarization"],
    )
    # add_label_ehva(c, die="demo_die", metadata_include_child=["width_mmi"])
    # add_label_ehva(c, die="demo_die", metadata_include_child=[])

    print(c.labels)
    c.show()
