from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.components.array import array
from pp.components.bend_euler import bend_euler
from pp.components.electrical.pad import pad
from pp.components.straight import straight
from pp.port import auto_rename_ports
from pp.types import ComponentOrFactory


@cell
def array_with_fanout(
    component: ComponentOrFactory = pad,
    n: int = 3,
    pitch: float = 150.0,
    waveguide_pitch: float = 10.0,
    start_straight: float = 5.0,
    end_straight: float = 40.0,
    radius: float = 5.0,
    component_port_name: str = "S",
    bend_port_name: str = "N0",
    bend_port_name_out: str = "W0",
    waveguide="metal_routing",
    **waveguide_settings,
) -> Component:
    """Returns an array of components in X axis
    with fanout waveguides facing west

    Args:
        component: to replicate
        n: number of components
        pitch: float
        waveguide_pitch: for fanout
        start_straight: length of the start of the straight
        end_straight: lenght of the straight at the end
        radius: bend radius
        waveguide: waveguide definition
        component_port_name:
        bend_port_name:
        bend_port_name_out:
        **waveguide_settings
    """
    c = Component()
    component = component() if callable(component) else component
    bend = bend_euler(radius=radius, waveguide=waveguide, **waveguide_settings)

    for col in range(n):
        ref = component.ref()
        ref.x = col * pitch
        c.add(ref)
        ylength = col * waveguide_pitch + start_straight
        xlength = col * pitch + end_straight
        straight_ref = c << straight(
            length=ylength, waveguide=waveguide, **waveguide_settings
        )
        straight_ref.connect("E0", ref.ports[component_port_name])

        bend_ref = c.add_ref(bend)
        bend_ref.connect(bend_port_name, straight_ref.ports["W0"])
        straightx_ref = c << straight(
            length=xlength, waveguide=waveguide, **waveguide_settings
        )
        straightx_ref.connect("E0", bend_ref.ports[bend_port_name_out])
        c.add_port(f"W_{col}", port=straightx_ref.ports["W0"])
    auto_rename_ports(c)
    return c


@cell
def array_with_fanout_2d(
    pitch: float = 150.0,
    pitch_x: Optional[float] = None,
    pitch_y: Optional[float] = None,
    cols: int = 3,
    rows: int = 2,
    **kwargs,
) -> Component:
    """Returns 2D array with fanout waveguides facing west.

    Args:
        pitch: 2D pitch
        pitch_x: defaults to pitch
        pitch_y: defaults to pitch
        cols:
        rows:
        kwargs:
            component: to replicate
            n: number of components
            pitch: float
            waveguide_pitch: for fanout
            start_straight: length of the start of the straight
            end_straight: lenght of the straight at the end
            radius: bend radius
            waveguide: waveguide definition
            component_port_name:
            bend_port_name:
            bend_port_name_out:
            **waveguide_settings
    """
    pitch_y = pitch_y or pitch
    pitch_x = pitch_x or pitch
    row = array_with_fanout(n=cols, pitch=pitch_x, **kwargs)
    return array(component=row, n=rows, pitch=pitch_y, axis="y")


if __name__ == "__main__":

    # c1 = pp.c.pad()
    # c2 = array(component=c1, pitch=150, n=2)
    # print(c2.ports.keys())

    c2 = array_with_fanout(n=3, width=10, radius=11, waveguide_pitch=20)
    c2.show(show_ports=True)
