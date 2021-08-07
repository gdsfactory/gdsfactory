from typing import Optional

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.port import auto_rename_ports


@cell
def array(
    component: gf.types.ComponentOrFactory = straight,
    n: int = 2,
    pitch: float = 20.0,
    axis: str = "y",
    rotation: int = 0,
    h_mirror: bool = False,
    v_mirror: bool = False,
) -> Component:
    """Returns an array of components.

    Args:
        component: to replicate
        n: number of components
        pitch: float
        axis: x or y
        rotation: in degrees
        h_mirror: horizontal mirror
        v_mirror: vertical mirror
    """
    c = Component()
    component = component() if callable(component) else component

    if axis not in ["x", "y"]:
        raise ValueError(f"Axis must be x or y, got {axis}")

    for i in range(n):
        ref = component.ref(rotation=rotation, h_mirror=h_mirror, v_mirror=v_mirror)
        c.aliases[i] = ref
        if axis == "x":
            ref.x = i * pitch
        else:
            ref.y = i * pitch
        c.add(ref)
        for port in ref.get_ports_list():
            c.add_port(f"{port.name}_{i}", port=port)
    auto_rename_ports(c)
    return c


@cell
def array_2d(
    pitch: float = 150.0,
    pitch_x: Optional[float] = None,
    pitch_y: Optional[float] = None,
    cols: int = 3,
    rows: int = 2,
    **kwargs,
) -> Component:
    pitch_y = pitch_y or pitch_x
    """Returns 2D array with fanout waveguides facing west.

    Args:
        picth: x and y pitch
        pitch_x: Optional X pitch
        pitch_y: Optional Y pitch
        cols:
        rows:
        kwargs:
            component: to replicate
            n: number of components
            pitch: float
            axis: x or y
            rotation: in degrees
            h_mirror: horizontal mirror
            v_mirror: vertical mirror

    """
    pitch_y = pitch_y or pitch
    pitch_x = pitch_x or pitch
    row = array(n=cols, pitch=pitch_x, axis="x", **kwargs)
    return array(component=row, n=rows, pitch=pitch_y, axis="y")


if __name__ == "__main__":

    # c1 = gf.components.pad()
    # c2 = array(component=c1, pitch=150, n=2)
    # print(c2.ports.keys())

    c2 = array()
    c2 = array_2d()
    c2.show(show_ports=True)
