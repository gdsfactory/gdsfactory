from typing import Tuple

import pydantic

import gdsfactory as gf
from gdsfactory.snap import snap_to_grid as snap
from gdsfactory.types import Layer, Strs


class Dft(pydantic.BaseModel):
    pad_size: Tuple[int, int] = (100, 100)
    pad_pitch: int = 125
    pad_width: int = 100
    pad_gc_spacing_opposed: int = 500
    pad_gc_spacing_adjacent: int = 1000


DFT = Dft()


port_types_all = ("vertical_te", "dc")
skip = ("cross_section", "decorator", "cross_section1", "cross_section2")


def add_label_ehva(
    component: gf.Component,
    die_name: str,
    port_types: Strs = port_types_all,
    layer: Layer = (66, 0),
) -> gf.Component:
    """Returns Component with test and measurement automated labels.

    Args:
        component: gdsfactory component
        die_name: string.
        port_types: list of port types to label.
        layer: text label layer.
    """

    t = f"""DIE NAME:{die_name}
COMPONENT NAME:{component.name}
"""
    if component.metadata_child:
        info = [
            f"COMPONENTINFO NAME: {k}, VALUE {v}\n"
            for k, v in component.metadata_child.changed.items()
            if k not in skip and isinstance(v, (str, int, float))
        ]
        t += "\n".join(info)

    if component.ports:
        for port_type in port_types:
            info_optical_ports = [
                f"OPTICALPORT TYPE: {port_type}, "
                f"POSITION RELATIVE:({snap(port.x)}, {snap(port.y)}),"
                f" ORIENTATION: {port.orientation}"
                for port in component.get_ports_list(port_type=port_type)
            ]
            t += "\n".join(info_optical_ports)

    component.unlock()
    label = gf.Label(
        text=t,
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
    c = gf.c.mmi2x2()
    c = gf.routing.add_fiber_array(c)
    add_label_ehva(c, die_name="demo_die")
    print(c.labels)
    c.show()
