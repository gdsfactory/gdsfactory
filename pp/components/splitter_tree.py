import pp
from pp.components import waveguide
from pp.routing.connect import connect_strip
from pp.ports.utils import get_ports_facing


@pp.autoname
def splitter_tree(
    coupler,
    n_o_outputs=4,
    bend_radius=10.0,
    separation=50,
    termination_component=waveguide(length=0.1),
):
    n_o_outputs = n_o_outputs
    c = pp.Component()

    coupler = pp.call_if_func(coupler)
    _coupler = c.add_ref(coupler)
    coupler_sep = coupler.ports["E1"].y - coupler.ports["E0"].y

    if n_o_outputs > 2:
        _cmp = splitter_tree(
            coupler=coupler,
            n_o_outputs=n_o_outputs // 2,
            bend_radius=bend_radius,
            separation=separation / 2,
        )
    else:
        _cmp = termination_component

    a = separation or _cmp.ports["W0"].y - _coupler.size_info.south
    if a < coupler_sep:
        tree_top = _cmp.ref(port_id="W0", position=_coupler.ports["E1"])
        tree_bot = _cmp.ref(
            port_id="W0", position=_coupler.ports["E0"], v_mirror=False  # True
        )

    else:
        d = 2 * bend_radius + 1
        a = max(a, d)

        tree_top = _cmp.ref(
            port_id="W0", position=_coupler.ports["E1"].position + (d, a)
        )
        tree_bot = _cmp.ref(
            port_id="W0",
            position=_coupler.ports["E0"].position + (d, -a),
            v_mirror=False,  # True,
        )

        c.add(connect_strip(coupler.ports["E1"], tree_top.ports["W0"]))
        c.add(connect_strip(coupler.ports["E0"], tree_bot.ports["W0"]))

    i = 0
    for p in get_ports_facing(tree_bot, "E"):
        c.add_port(name="{}".format(i), port=p)
        i += 1

    for p in get_ports_facing(tree_top, "E"):
        c.add_port(name="{}".format(i), port=p)
        i += 1

    c.add(tree_bot)
    c.add(tree_top)
    c.add_port(name="W0", port=_coupler.ports["W0"])

    return c


if __name__ == "__main__":
    c = splitter_tree(coupler=pp.c.mmi1x2())
    pp.show(c)
