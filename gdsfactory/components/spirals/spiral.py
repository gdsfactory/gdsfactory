from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec


@gf.cell
def spiral(
    length: float = 100,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: ComponentSpec = "strip",
    spacing: float = 3.0,
    n_loops: int = 6,
) -> gf.Component:
    """Returns a spiral double (spiral in, and then out).

    Args:
        length: length of the spiral straight section.
        bend: bend component.
        straight: straight component.
        cross_section: cross_section component.
        spacing: spacing between the spiral loops.
        n_loops: number of loops.
    """
    c = gf.Component()
    b = gf.get_component(bend, cross_section=cross_section)

    o1 = b["o1"]
    o2 = b["o2"]
    dx = abs(o2.dx - o1.dx)
    dy = abs(o2.dy - o1.dy)

    if dx != dy:
        raise ValueError(f"bend component {b} must have dx == dy")
    radius = dx
    _length = length

    b_inners = [c << b for _ in range(4)]
    b_inners[0].dmirror()
    b_inners[1].connect("o1", b_inners[0], "o2")
    b_inners[2].connect("o1", b_inners[1], "o2")
    s_space = c << gf.get_component(
        straight, cross_section=cross_section, length=spacing
    )
    s_space.connect("o1", b_inners[2], "o2")
    b_inners[3].connect("o1", s_space, "o2")
    l0_2 = c << gf.get_component(
        straight, cross_section=cross_section, length=_length + 2 * radius + spacing
    )
    l0_2.connect("o1", b_inners[3], "o2")
    p2 = l0_2.ports["o2"]

    if length > 0:
        l0_1 = c << gf.get_component(
            straight, cross_section=cross_section, length=_length
        )
        l0_1.connect("o1", b_inners[0], "o1")
        p1 = l0_1.ports["o2"].copy()
    else:
        p1 = b_inners[0].ports["o1"]
    p1.dmirror = not p1.dmirror
    for i in range(n_loops // 2):
        bends = [c << b for _ in range(8)]
        bends[0].connect("o1", p1)
        bends[1].connect("o1", p2)
        v1 = c << gf.get_component(
            straight, cross_section=cross_section, length=spacing * (1 + 4 * i)
        )
        v1.connect("o1", bends[0], "o2")
        v2 = c << gf.get_component(
            straight, cross_section=cross_section, length=spacing * (3 + 4 * i)
        )
        v2.connect("o1", bends[1], "o2")
        bends[2].connect("o1", v1, "o2")
        bends[3].connect("o1", v2, "o2")
        h1 = c << gf.get_component(
            straight,
            cross_section=cross_section,
            length=_length + 2 * radius + spacing * (1 + 4 * i),
        )
        h2 = c << gf.get_component(
            straight,
            cross_section=cross_section,
            length=_length + 2 * radius + spacing * (3 + 4 * i),
        )
        h1.connect("o1", bends[2], "o2")
        h2.connect("o1", bends[3], "o2")
        bends[4].connect("o1", h1, "o2")
        bends[5].connect("o1", h2, "o2")
        v3 = c << gf.get_component(
            straight, cross_section=cross_section, length=spacing * (3 + 4 * i)
        )
        v4 = c << gf.get_component(
            straight, cross_section=cross_section, length=spacing * (5 + 4 * i)
        )
        v3.connect("o1", bends[4], "o2")
        v4.connect("o1", bends[5], "o2")
        bends[6].connect("o1", v3, "o2")
        bends[7].connect("o1", v4, "o2")
        h3 = c << gf.get_component(
            straight,
            cross_section=cross_section,
            length=_length + 2 * radius + spacing * (3 + 4 * i),
        )
        h4 = c << gf.get_component(
            straight,
            cross_section=cross_section,
            length=_length + 2 * radius + spacing * (5 + 4 * i),
        )
        h3.connect("o1", bends[6], "o2")
        h4.connect("o1", bends[7], "o2")
        p1 = h3.ports["o2"]
        p2 = h4.ports["o2"]

    c.add_port(name="o1", port=p1)
    c.add_port(name="o2", port=p2)
    c.info["length"] = length * 2 * n_loops
    return c


if __name__ == "__main__":
    c = spiral(cross_section="rib", length=10, spacing=3.0)
    c.show()
