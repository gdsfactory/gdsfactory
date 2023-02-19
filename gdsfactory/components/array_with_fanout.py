from __future__ import annotations

from typing import Optional

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.array_component import array
from gdsfactory.components.straight import straight
from gdsfactory.port import auto_rename_ports
from gdsfactory.routing.sort_ports import sort_ports_x
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@cell
def array_with_fanout(
    component: ComponentSpec = "pad",
    columns: int = 3,
    pitch: float = 150.0,
    waveguide_pitch: float = 10.0,
    start_straight_length: float = 5.0,
    end_straight_length: float = 40.0,
    radius: float = 5.0,
    component_port_name: str = "e4",
    bend: ComponentSpec = "bend_euler",
    bend_port_name1: Optional[str] = None,
    bend_port_name2: Optional[str] = None,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns component array in X axis with west facing waveguides.

    Args:
        component: to replicate.
        columns: number of components.
        pitch: for waveguides.
        waveguide_pitch: for output waveguides.
        start_straight_length: length of the start of the straight.
        end_straight_length: length of the straight at the end.
        radius: bend radius.
        component_port_name: for fanout.
        bend: spec.
        bend_port_name1: optional port name.
        bend_port_name2: optional port name.
        cross_section: cross_section spec.
        kwargs: cross_section settings.
    """
    c = Component()
    component = gf.get_component(component)
    bend = gf.get_component(bend, radius=radius, cross_section=cross_section, **kwargs)

    bend_ports = bend.get_ports_list()
    bend_ports = sort_ports_x(bend_ports)
    bend_ports.reverse()
    bend_port_name1 = bend_port_name1 or bend_ports[0].name
    bend_port_name2 = bend_port_name2 or bend_ports[1].name

    for col in range(columns):
        ref = component.ref()
        ref.x = col * pitch
        c.add(ref)
        ylength = col * waveguide_pitch + start_straight_length
        xlength = col * pitch + end_straight_length
        straight_ref = c << straight(
            length=ylength, cross_section=cross_section, **kwargs
        )
        port_s1, port_s2 = straight_ref.get_ports_list()

        straight_ref.connect(port_s2.name, ref.ports[component_port_name])

        bend_ref = c.add_ref(bend)
        bend_ref.connect(bend_port_name1, straight_ref.ports[port_s1.name])
        straightx_ref = c << straight(
            length=xlength, cross_section=cross_section, **kwargs
        )
        straightx_ref.connect(port_s2.name, bend_ref.ports[bend_port_name2])
        c.add_port(f"W_{col}", port=straightx_ref.ports[port_s1.name])
    auto_rename_ports(c)
    return c


@cell
def array_with_fanout_2d(
    pitch: float = 150.0,
    pitch_x: Optional[float] = None,
    pitch_y: Optional[float] = None,
    columns: int = 3,
    rows: int = 2,
    **kwargs,
) -> Component:
    """Returns 2D array with fanout waveguides facing west.

    Args:
        pitch: 2D pitch.
        pitch_x: defaults to pitch.
        pitch_y: defaults to pitch.
        columns: number of columns.
        rows: number of rows.

    keyword args:
        component: to replicate.
        pitch: in um.
        waveguide_pitch: for fanout in um.
        start_straight_length: length of the start of the straight in um.
        end_straight_length: length of the straight at the end in um.
        radius: bend radius in um.
        cross_section: cross_section factory.
        component_port_name:
        bend_port_name1:
        bend_port_name2:
    """
    pitch_y = pitch_y or pitch
    pitch_x = pitch_x or pitch
    row = array_with_fanout(columns=columns, pitch=pitch_x, **kwargs)
    return array(component=row, rows=rows, columns=1, spacing=(0, pitch_y))


def test_array_with_fanout() -> None:
    c1 = array_with_fanout_2d(columns=2, rows=2)
    assert len(c1.ports) == 4


def test_array() -> None:
    c1 = array_with_fanout_2d(columns=2, rows=2)
    assert len(c1.ports) == 4


if __name__ == "__main__":
    from gdsfactory.generic_tech import get_generic_pdk

    PDK = get_generic_pdk()
    PDK.activate()
    # import gdsfactory as gf
    # c1 = gf.components.pad()
    # c2 = array(component=c1, pitch=150, columns=2)
    # print(c2.ports.keys())
    # c = array_with_fanout(
    #     columns=3,
    #     waveguide_pitch=20,
    #     bend=gf.components.wire_corner,
    #     cross_section='metal_routing',
    #     layer=(2, 0),
    #     width=10,
    #     radius=11,
    # )
    c = array_with_fanout_2d()
    c.show(show_ports=True)
