from typing import Optional

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.array import array
from gdsfactory.components.electrical.pad import pad
from gdsfactory.components.straight import straight
from gdsfactory.components.via_stack import via_stack
from gdsfactory.cross_section import metal2
from gdsfactory.port import auto_rename_ports
from gdsfactory.types import ComponentFactory, ComponentOrFactory, CrossSectionFactory


@cell
def array_with_via(
    component: ComponentOrFactory = pad,
    n: int = 3,
    pitch: float = 150.0,
    waveguide_pitch: float = 10.0,
    end_straight: float = 60.0,
    radius: float = 5.0,
    component_port_name: str = "S",
    bend_port_name: str = "N0",
    cross_section: CrossSectionFactory = metal2,
    via_stack: ComponentFactory = via_stack,
    via_stack_y_offset: float = -44.0,
    **kwargs,
) -> Component:
    """Returns an array of components in X axis
    with fanout waveguides facing west

    Args:
        component: to replicate
        n: number of components
        pitch: float
        waveguide_pitch: for fanout
        end_straight: lenght of the straight at the end
        radius: bend radius
        waveguide: waveguide definition
        component_port_name:
        bend_port_name:
        via_stack_port_name:
        **kwargs
    """
    c = Component()
    component = component() if callable(component) else component
    via_stack = via_stack()

    for col in range(n):
        ref = component.ref()
        ref.x = col * pitch
        c.add(ref)
        xlength = col * pitch + end_straight

        via_stack_ref = c << via_stack
        via_stack_ref.x = col * pitch
        via_stack_ref.y = col * waveguide_pitch + via_stack_y_offset

        straightx_ref = c << straight(
            length=xlength, cross_section=cross_section, **kwargs
        )
        straightx_ref.connect("E0", via_stack_ref.ports["W0"])
        c.add_port(f"W_{col}", port=straightx_ref.ports["W0"])
    auto_rename_ports(c)
    return c


@cell
def array_with_via_2d(
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
            end_straight: lenght of the straight at the end
            radius: bend radius
            component_port_name:
            bend_port_name:
            via_stack_port_name:
            **kwargs
    """
    pitch_y = pitch_y or pitch
    pitch_x = pitch_x or pitch
    row = array_with_via(n=cols, pitch=pitch_x, **kwargs)
    return array(component=row, n=rows, pitch=pitch_y, axis="y")


if __name__ == "__main__":
    # c2 = array_with_via(n=3, width=10, radius=11, waveguide_pitch=20)
    cols = rows = 8
    c2 = array_with_via_2d(cols=cols, rows=rows, waveguide_pitch=12)
    c2.show()
