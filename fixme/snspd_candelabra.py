"""fixme."""

import numpy as np
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.types import LayerSpec
from gdsfactory.components.optimal_hairpin import optimal_hairpin
from gdsfactory.components.optimal_90deg import optimal_90deg
from gdsfactory.components.compass import compass
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.geometry.boolean import boolean


@cell
def snspd_candelabra(
    wire_width=0.52,
    wire_pitch=0.56,
    haxis=90,
    vaxis=50,
    equalize_path_lengths=False,
    xwing=False,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Returns optimally-rounded SNSPD with low current crowding and \
            arbitrarily-high fill factor.

    described by Reddy et. al.,
    APL Photonics 7, 051302 (2022)  https://doi.org/10.1063/5.0088007

    Args:
        wire_width : int or float
            Width of the wire.
        wire_pitch : int or float
            Distance between two adjacent wires. Must be greater than `width`.
        haxis : int or float
            Length of horizontal diagonal of the rhomboidal active area.
            The parameter `haxis` is prioritized over `vaxis`.
        vaxis : int or float
            Length of vertical diagonal of the rhomboidal active area.
        equalize_path_lengths : bool
            If True, adds wire segments to hairpin bends to equalize path lengths
            from center to center for all parallel wires in active area.
        xwing : bool
            If True, replaces 90-degree bends with 135-degree bends.
        layer: layer spec to put polygon geometry on.

    """

    def off_axis_uturn(
        wire_width=0.52,
        wire_pitch=0.56,
        pfact=10.0 / 3,
        sharp=False,
        pad_length=0,
        layer=0,
    ):
        """Returns Component low-crowding u-turn for candelabra meander."""
        barc = optimal_90deg(width=wire_width, layer=layer)
        if not sharp:
            # For non-rounded outer radii
            # Not fully implemented
            port1mp = [barc.ports["e1"].x, barc.ports["e1"].y]
            port1or = barc.ports["e1"].orientation
            port2mp = [barc.ports["e2"].x, barc.ports["e2"].y]
            port2or = barc.ports["e2"].orientation
            barc = boolean(
                A=barc,
                B=barc.copy().move([-wire_width, -wire_width]),
                operation="not",
                layer=layer,
            )
            barc.add_port(
                name="o1",
                center=port1mp,
                width=wire_width,
                orientation=port1or,
                layer=layer,
            )
            barc.add_port(
                name="o2",
                center=port2mp,
                width=wire_width,
                orientation=port2or,
                layer=layer,
            )
        pin = optimal_hairpin(
            width=wire_width,
            pitch=pfact * wire_width,
            length=8 * wire_width,
            layer=layer,
        )
        pas = compass(size=(wire_width, wire_pitch), layer=layer)
        D = Component()
        arc1 = D.add_ref(barc)
        arc1.rotate(90)
        pin1 = D.add_ref(pin)
        pin1.connect("e1", arc1.ports["o2"])
        pas1 = D.add_ref(pas)
        pas1.connect(pas1.ports["e2"], pin1.ports["e2"])
        arc2 = D.add_ref(barc)
        arc2.connect("o2", pas1.ports["e4"])
        if pad_length > 0:
            pin1.movey(pad_length * 0.5)
            tempc = D.add_ref(
                compass(
                    size=(
                        pin1.ports["e1"].width,
                        pin1.ports["e1"].y - arc1.ports["o2"].y,
                    ),
                    layer=layer,
                )
            )
            tempc.connect("N", pin1.ports["e1"])
            tempc = D.add_ref(
                compass(
                    size=(pin1.ports["e1"].width, pin1.ports[2].y - pas1.ports["e2"].y),
                    layer=layer,
                )
            )
            tempc.connect("e2", pin1.ports["e2"])
        D.add_port(
            name="o1",
            center=arc1.ports["o1"].center,
            width=wire_width,
            orientation=arc1.ports["o1"].orientation,
            layer=layer,
        )
        D.add_port(
            name="o2",
            center=arc2.ports["o1"].center,
            width=wire_width,
            orientation=arc2.ports["o1"].orientation,
            layer=layer,
        )
        return D

    def xwing_uturn(
        wire_width=0.52, wire_pitch=0.56, pfact=10.0 / 3, pad_length=0, layer=0
    ):
        """Returns Component low-crowding u-turn for X-wing meander."""
        barc = bend_circular(
            radius=wire_width * 3, width=wire_width, layer=layer, theta=45
        ).rotate(180)

        pin = optimal_hairpin(
            width=wire_width,
            pitch=pfact * wire_width,
            length=15 * wire_width,
            layer=layer,
        )

        paslen = pfact * wire_width - np.sqrt(2) * wire_pitch
        pas = compass(size=(wire_width, abs(paslen)), layer=layer)
        Dtemp = Component()
        arc1 = Dtemp.add_ref(barc)
        arc1.rotate(90)
        pin1 = Dtemp.add_ref(pin)
        pas1 = Dtemp.add_ref(pas)
        arc2 = Dtemp.add_ref(barc)
        if paslen > 0:
            pas1.connect(pas1.ports["e3"], arc1.ports[2])
            pin1.connect(1, pas1.ports["e2"])
            arc2.connect(2, pin1.ports[2])
        else:
            pin1.connect(1, arc1.ports[2])
            pas1.connect("e2", pin1.ports[2])
            arc2.connect(2, pas1.ports["e4"])
        if pad_length > 0:
            pin1.move([pad_length * 0.5 / np.sqrt(2), pad_length * 0.5 / np.sqrt(2)])
            if paslen > 0:
                indx1 = 2
                indx2 = 1
                myarc = arc2
            else:
                indx1 = 1
                indx2 = 2
                myarc = arc1
            compdist = np.sqrt(
                np.sum(np.square(pin1.ports[indx1].center - myarc.ports[2].center))
            )
            tempc = Dtemp.add_ref(
                compass(size=(pin1.ports[indx1].width, compdist), layer=layer)
            )
            tempc.connect("N", pin1.ports[indx1])
            compdist = np.sqrt(
                np.sum(np.square(pin1.ports[indx2].center - pas1.ports["N"].center))
            )
            tempc = Dtemp.add_ref(
                compass(size=(pin1.ports[indx2].width, compdist), layer=layer)
            )
            tempc.connect("N", pin1.ports[indx2])

        Dtemp.add_port(
            name=1,
            center=arc1.ports["o1"].center,
            width=wire_width,
            orientation=arc1.ports["o1"].orientation,
            layer=layer,
        )
        Dtemp.add_port(
            name=2,
            center=arc2.ports["o1"].center,
            width=wire_width,
            orientation=arc2.ports["o1"].orientation,
            layer=layer,
        )

        return Dtemp

    D = Component(name="snspd_candelabra")
    if xwing:
        Dtemp = xwing_uturn(wire_width=wire_width, wire_pitch=wire_pitch, layer=layer)
    else:
        Dtemp = off_axis_uturn(
            wire_width=wire_width, wire_pitch=wire_pitch, layer=layer
        )
    Dtemp_mirrored = Dtemp.copy().mirror([0, 0], [0, 1])
    padding = Dtemp.xsize
    maxll = haxis - 2 * padding
    dll = abs(Dtemp.ports[1].x - Dtemp.ports[2].x) + wire_pitch
    half_num_meanders = int(np.ceil(0.5 * vaxis / wire_pitch)) + 2

    if xwing:
        bend = D.add_ref(
            bend_circular(
                radius=wire_width * 3, width=wire_width, theta=90, layer=layer
            )
        ).rotate(180)
    else:
        bend = D.add_ref(optimal_90deg(width=wire_width, layer=layer))
    if (maxll - dll * half_num_meanders) <= 0.0:
        while (maxll - dll * half_num_meanders) <= 0.0:
            half_num_meanders = half_num_meanders - 1
    fpas = D.add_ref(
        compass(size=(0.5 * (maxll - dll * half_num_meanders), wire_width), layer=layer)
    )
    D.movex(-bend.ports[1].x)
    fpas.connect(fpas.ports["W"], bend.ports[2])
    ll = D.xsize * 2 - wire_width
    if equalize_path_lengths:
        if xwing:
            Dtemp = xwing_uturn(
                wire_width=wire_width,
                wire_pitch=wire_pitch,
                pad_length=(maxll - ll - dll) * equalize_path_lengths,
                layer=layer,
            )
        else:
            Dtemp = off_axis_uturn(
                wire_width=wire_width,
                wire_pitch=wire_pitch,
                pad_length=(maxll - ll - dll) * equalize_path_lengths,
                layer=layer,
            )
    uturn = D.add_ref(Dtemp)
    uturn.connect(1, fpas.ports["E"])
    dir_left = True

    turn_padding = maxll - ll - 2 * dll

    while ll < maxll - dll:
        ll = ll + dll
        if equalize_path_lengths:
            if xwing:
                Dtemp = xwing_uturn(
                    wire_width=wire_width,
                    wire_pitch=wire_pitch,
                    pad_length=turn_padding * equalize_path_lengths,
                    layer=layer,
                )
            else:
                Dtemp = off_axis_uturn(
                    wire_width=wire_width,
                    wire_pitch=wire_pitch,
                    pad_length=turn_padding * equalize_path_lengths,
                    layer=layer,
                )
        turn_padding = turn_padding - dll
        newpas = D.add_ref(compass(size=(ll, wire_width), layer=layer))
        if dir_left:
            newpas.connect(newpas.ports["E"], uturn.ports[2])
            if equalize_path_lengths:
                uturn = D.add_ref(Dtemp.mirror([0, 0], [0, 1]))
            else:
                uturn = D.add_ref(Dtemp_mirrored)
            uturn.connect(1, newpas.ports["W"])
            dir_left = False
        else:
            newpas.connect(newpas.ports["W"], uturn.ports[2])
            uturn = D.add_ref(Dtemp)
            uturn.connect(1, newpas.ports["E"])
            dir_left = True

    newpas = D.add_ref(compass(size=(ll / 2, wire_width), layer=layer))
    if dir_left:
        newpas.connect(newpas.ports["E"], uturn.ports[2])
        dir_left = False
    else:
        newpas.connect(newpas.ports["W"], uturn.ports[2])
        dir_left = True

    D.movex(-D.x)
    if not xwing:
        bend.movex(-bend.ports[1].x)
    if (fpas.ports["W"].x - bend.ports[2].x) > 0:
        tempc = D.add_ref(
            compass(
                size=(fpas.ports["W"].x - bend.ports[2].x, bend.ports[2].width),
                layer=layer,
            )
        )
        tempc.connect("E", fpas.ports["W"])
    D.move([-D.x, -D.ymin - wire_width * 0.5])
    D.add_port(name=1, port=bend.ports[1])
    if dir_left:
        D.add_port(name=2, port=newpas.ports["E"])
    else:
        D.add_port(name=2, port=newpas.ports["W"])

    out = Component()
    D1 = out.add_ref(D)
    D2 = out.add_ref(D.rotate(180))
    tempc = out.add_ref(
        compass(
            size=(abs(D1.ports[2].x - D2.ports[2].x), D1.ports[2].width), layer=layer
        )
    )
    if D1.ports[2].x > D2.ports[2].x:
        tempc.connect("E", D1.ports[2])
    else:
        tempc.connect("W", D1.ports[2])
    out.add_port(name=1, port=D1.ports[1])
    out.add_port(name=2, port=D2.ports[1])
    return out


if __name__ == "__main__":
    c = snspd_candelabra()
    c.show(show_ports=True)
