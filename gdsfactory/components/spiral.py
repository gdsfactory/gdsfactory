from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.typings import ComponentSpec


@gf.cell
def spiral(
    length: float = 100,
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    cross_section: ComponentSpec = "xs_sc",
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
    xs = gf.get_cross_section(cross_section)

    b = bend(cross_section=cross_section)
    radius = xs.radius
    _length = length

    s_f = partial(straight, cross_section=cross_section)
    b_inners = [c << b for _ in range(4)]
    b_inners[1].connect("o2", b_inners[0], "o2")
    b_inners[2].connect("o1", b_inners[1], "o1", mirror=True)
    s_space = c << s_f(length=spacing)
    s_space.connect("o1", b_inners[2], "o2")
    b_inners[3].connect("o1", s_space, "o2")
    l0_2 = c << s_f(length=_length + 2 * radius + spacing)
    l0_2.connect("o1", b_inners[3], "o2")
    p2 = l0_2.ports["o2"]

    if length > 0:
        l0_1 = c << s_f(length=_length)
        l0_1.connect("o1", b_inners[0], "o1")
        p1 = l0_1.ports["o2"].copy()
        p1.mirror = not p1.mirror
    else:
        p1 = b_inners[0].ports["o1"]
        p1.mirror = not p1.mirror

    for i in range(n_loops // 2):
        bends = [c << b for _ in range(8)]
        bends[0].connect("o1", p1)
        bends[1].connect("o1", p2)
        v1 = c << s_f(length=spacing * (1 + 4 * i))
        v1.connect("o1", bends[0], "o2")
        v2 = c << s_f(length=spacing * (3 + 4 * i))
        v2.connect("o1", bends[1], "o2")
        bends[2].connect("o1", v1, "o2")
        bends[3].connect("o1", v2, "o2")
        h1 = c << s_f(length=_length + 2 * radius + spacing * (1 + 4 * i))
        h2 = c << s_f(length=_length + 2 * radius + spacing * (3 + 4 * i))
        h1.connect("o1", bends[2], "o2")
        h2.connect("o1", bends[3], "o2")
        bends[4].connect("o1", h1, "o2")
        bends[5].connect("o1", h2, "o2")
        v3 = c << s_f(length=spacing * (3 + 4 * i))
        v4 = c << s_f(length=spacing * (5 + 4 * i))
        v3.connect("o1", bends[4], "o2")
        v4.connect("o1", bends[5], "o2")
        bends[6].connect("o1", v3, "o2")
        bends[7].connect("o1", v4, "o2")
        h3 = c << s_f(length=_length + 2 * radius + spacing * (3 + 4 * i))
        h4 = c << s_f(length=_length + 2 * radius + spacing * (5 + 4 * i))
        h3.connect("o1", bends[6], "o2")
        h4.connect("o1", bends[7], "o2")
        p1 = h3.ports["o2"]
        p2 = h4.ports["o2"]

    c.add_port(name="o1", port=p1)
    c.add_port(name="o2", port=p2)
    return c


if __name__ == "__main__":
    c = spiral(cross_section="xs_rc", length=10, spacing=3.0)
    c.show()
