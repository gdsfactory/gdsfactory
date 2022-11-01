import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.types import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_single_sample(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.010,
    coupler_ring: ComponentSpec = "coupler_ring",
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    cross_section: CrossSectionSpec = "strip",
    **kwargs
) -> Component:
    """Single bus ring made of a ring coupler.

    (cb: bottom) connected with two vertical straights (wl: left, wr: right)
    two bends (bl, br) and horizontal straight (wg: top).

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler function.
        straight: straight function.
        bend: 90 degrees bend function.
        cross_section: spec.
        kwargs: cross_section settings.


    .. code::

          bl-wt-br
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x

    """
    gap = gf.snap.snap_to_grid(gap, nm=2)

    coupler_ring_component = gf.get_component(
        coupler_ring,
        bend=bend,
        gap=gap,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        **kwargs
    )
    straight_side = gf.get_component(
        straight, length=length_y, cross_section=cross_section, **kwargs
    )
    straight_top = gf.get_component(
        straight, length=length_x, cross_section=cross_section, **kwargs
    )

    bend = gf.get_component(bend, radius=radius, cross_section=cross_section, **kwargs)

    c = Component()
    cb = c << coupler_ring_component
    wl = c << straight_side
    wr = c << straight_side
    bl = c << bend
    br = c << bend
    wt = c << straight_top
    # wt.mirror(p1=(0, 0), p2=(1, 0))

    wl.connect(port="o2", destination=cb.ports["o2"])
    bl.connect(port="o2", destination=wl.ports["o1"])

    wt.connect(port="o2", destination=bl.ports["o1"])
    br.connect(port="o2", destination=wt.ports["o1"])
    wr.connect(port="o1", destination=br.ports["o1"])

    c.add_port("o2", port=cb.ports["o2"])
    c.add_port("o1", port=cb.ports["o1"])
    return c


def test_ring_single_sample() -> None:
    assert ring_single_sample()


if __name__ == "__main__":
    c = ring_single_sample(width=2, gap=1, layer=(2, 0))
    print(c.ports)
    c.show(show_ports=True)
