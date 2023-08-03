from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.components.bend_s import bend_s as bend_s_function
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.mmi2x2 import mmi2x2
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Float2


@gf.cell
def splitter_tree(
    coupler: ComponentSpec = mmi1x2,
    noutputs: int = 4,
    spacing: Float2 = (90.0, 50.0),
    bend_s: ComponentSpec | None = bend_s_function,
    bend_s_xsize: float | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Tree of power splitters.

    Args:
        coupler: coupler factory.
        noutputs: number of outputs.
        spacing: x, y spacing between couplers.
        bend_s: Sbend function for termination.
        bend_s_xsize: xsize for the sbend.
        cross_section: cross_section.

    .. code::

             __|
          __|  |__
        _|  |__
         |__        dy

          dx
    """
    c = gf.Component()

    dx, dy = spacing

    coupler = gf.get_component(coupler)
    coupler_ports_west = coupler.get_ports_list(port_type="optical", orientation=180)
    coupler_ports_east = coupler.get_ports_list(port_type="optical", orientation=0)

    e1_port_name = coupler_ports_east[0].name
    e0_port_name = coupler_ports_east[1].name
    w0_port_name = coupler_ports_west[0].name

    if bend_s:
        dy_coupler_ports = abs(
            coupler.ports[e0_port_name].center[1]
            - coupler.ports[e1_port_name].center[1]
        )
        bend_s_ysize = dy / 4 - dy_coupler_ports / 2
        bend_s_xsize = bend_s_xsize or dx
        bend_s = gf.get_component(
            bend_s,
            cross_section=cross_section,
            size=(bend_s_xsize, bend_s_ysize),
        )
        c.info["bend_s"] = bend_s.info
    cols = int(np.log2(noutputs))
    i = 0

    for col in range(cols):
        ncouplers = int(2**col)
        y0 = -0.5 * dy * 2 ** (cols - 1)
        for row in range(ncouplers):
            x = col * dx
            y = y0 + (row + 0.5) * dy * 2 ** (cols - col - 1)
            coupler_ref = c.add_ref(coupler, alias=f"coupler_{col}_{row}")
            coupler_ref.move((x, y))
            if col == 0:
                for port in coupler_ref.get_ports_list():
                    if port.name not in [e0_port_name, e1_port_name]:
                        c.add_port(name=f"{port.name}_{col}_{i}", port=port)
                        i += 1
            if col > 0:
                if row % 2 == 0:
                    port_name = e0_port_name
                if row % 2 == 1:
                    port_name = e1_port_name
                c.add(
                    gf.routing.get_route(
                        c.named_references[f"coupler_{col-1}_{row//2}"].ports[
                            port_name
                        ],
                        coupler_ref.ports["o1"],
                        cross_section=cross_section,
                    ).references
                )
            if cols > col > 0:
                for port in coupler_ref.get_ports_list():
                    if port.name not in [
                        "o1",
                        e0_port_name,
                        e1_port_name,
                        w0_port_name,
                    ]:
                        c.add_port(name=f"{port.name}_{col}_{i}", port=port)
                        i += 1
            if col == cols - 1 and bend_s is None:
                for port in coupler_ref.get_ports_list():
                    if port.name in [e1_port_name, e0_port_name]:
                        c.add_port(name=f"{port.name}_{col}_{i}", port=port)
                        i += 1
            if col == cols - 1 and bend_s:
                btop = c << bend_s
                bbot = c << bend_s
                bbot.mirror()
                btop.connect("o1", coupler_ref.ports[e1_port_name])
                bbot.connect("o1", coupler_ref.ports[e0_port_name])
                port = btop.ports["o2"]
                c.add_port(name=f"{port.name}_{col}_{i}", port=port)
                i += 1
                port = bbot.ports["o2"]
                c.add_port(name=f"{port.name}_{col}_{i}", port=port)
                i += 1

    return c


def test_splitter_tree_ports() -> None:
    c = splitter_tree(
        coupler=mmi2x2,
        noutputs=4,
    )
    assert len(c.ports) == 8, len(c.ports)


def test_splitter_tree_ports_no_sbend() -> None:
    c = splitter_tree(
        coupler=mmi2x2,
        noutputs=4,
        bend_s=None,
    )
    assert len(c.ports) == 8, len(c.ports)


if __name__ == "__main__":
    test_splitter_tree_ports()
    test_splitter_tree_ports_no_sbend()

    import gdsfactory as gf

    # c = splitter_tree(
    #     coupler=partial(mmi2x2, gap_mmi=2.0, width_mmi=5.0),
    #     # noutputs=128 * 2,
    #     # noutputs=2 ** 3,
    #     noutputs=2**2,
    #     # bend_s=None,
    #     # dy=100.0,
    #     # layer=(2, 0),
    # )
    c = splitter_tree(
        noutputs=2**2,
        spacing=(120.0, 50.0),
        # bend_length=30,
        # bend_s=None,
        cross_section="rib_conformal2",
    )
    c.show(show_ports=True)
    # print(len(c.ports))
    # for port in c.get_ports_list():
    #     print(port)
