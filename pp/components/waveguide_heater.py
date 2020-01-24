import numpy as np
from pp import Component
import pp
from pp.name import autoname
from pp.layers import LAYER
from pp.ports import deco_rename_ports
from pp.components.waveguide import waveguide
from pp.components.hline import hline
from pp.components.electrical.tlm import tlm
from pp.components.extension import line


__version__ = "0.0.1"


@deco_rename_ports
@autoname
def heater(length=10, width=0.5, layers_heater=[LAYER.HEATER]):
    """ straight heater
    """
    c = pp.Component()
    for layer in layers_heater:
        _ref = c.add_ref(hline(length=length, width=width, layer=layer))
        c.ports = _ref.ports  # Use ports from latest layer as heater ports
        c.absorb(_ref)
    return c


def add_trenches(
    c,
    sstw=2.0,
    trench_width=0.5,
    trench_keep_out=2.0,
    trenches=[
        {"nb_segments": 2, "lane": 1, "x_start_offset": 0},
        {"nb_segments": 2, "lane": -1, "x_start_offset": 0},
    ],
    layer_trench=LAYER.DEEPTRENCH,
):
    """
    Add trenches to a waveguide-heater-like component
    """

    heater_width = c.settings["heater_width"]
    heater_spacing = c.settings["heater_spacing"]
    width = c.settings["width"]
    length = c.settings["length"]

    a = heater_spacing + (width + heater_width) / 2

    # Add trenches
    if trench_width and trench_width > 0:
        tko = trench_keep_out

        for trench in trenches:
            lane = trench["lane"]
            td = tko + a + (trench_width + heater_width) / 2
            y = np.sign(lane) * (td + (abs(lane) - 1) * (trench_width + tko))
            x_start_offset = trench["x_start_offset"]

            if "segments" not in trench:
                nb_segments = trench["nb_segments"]
                trench_length = (length - (nb_segments - 1) * sstw) / nb_segments
                segments = [trench_length] * nb_segments
            else:
                segments = trench["segments"]
            x = x_start_offset
            for i, trench_length in enumerate(segments):
                trench = hline(
                    length=trench_length, width=trench_width, layer=layer_trench
                )
                _trench = trench.ref(
                    port_id="W0", position=c.ports["W0"].position + (x, y)
                )
                c.add(_trench)
                c.absorb(_trench)
                x += trench_length + sstw

    return c


@deco_rename_ports
@autoname
def waveguide_heater(
    length=10.0,
    width=0.5,
    heater_width=0.5,
    heater_spacing=1.2,
    metal_connection=True,
    sstw=2.0,
    trench_width=0.5,
    trench_keep_out=2.0,
    trenches=[
        {"nb_segments": 2, "lane": 1, "x_start_offset": 0},
        {"nb_segments": 2, "lane": -1, "x_start_offset": 0},
    ],
    layers_heater=[LAYER.HEATER],
    waveguide_factory=waveguide,
    layer_trench=LAYER.DEEPTRENCH,
):
    """ waveguide with heater

    .. code::
    
        TTTTTTTTTTTTT    TTTTTTTTTTTTT <-- trench
        
        HHHHHHHHHHHHHHHHHHHHHHHHHHHHHH <-- heater
        
        ------------------------------ <-- waveguide

        HHHHHHHHHHHHHHHHHHHHHHHHHHHHHH <-- heater
        
        TTTTTTTTTTTTT    TTTTTTTTTTTTT <-- trench

    .. plot::
      :include-source:

      import pp

      c = pp.c.waveguide_heater()
      pp.plotgds(c)
    
    """
    c = Component()

    _heater = heater(length=length, width=heater_width, layers_heater=layers_heater)

    y_heater = heater_spacing + (width + heater_width) / 2
    heater_top = c << _heater
    heater_bot = c << _heater

    heater_top.movey(+y_heater)
    heater_bot.movey(-y_heater)

    wg = c << waveguide_factory(length=length, width=width)

    for i in [heater_top, heater_bot, wg]:
        c.absorb(i)

    # Add wg ports
    for p in wg.ports.values():
        c.add_port(name=p.name, port=p)

    # Add heater ports
    for p in heater_top.ports.values():
        c.add_port(name="HT" + p.name, port=p)

    for p in heater_bot.ports.values():
        c.add_port(name="HB" + p.name, port=p)

    c.settings["width"] = width
    c.settings["heater_width"] = heater_width
    c.settings["heater_spacing"] = heater_spacing
    c.settings["length"] = length
    add_trenches(
        c, sstw, trench_width, trench_keep_out, trenches, layer_trench=layer_trench
    )

    return c


@autoname
def wg_heater_connector(
    heater_ports,
    metal_width=10.0,
    tlm_layers=[LAYER.VIA1, LAYER.M1, LAYER.VIA2, LAYER.M2, LAYER.VIA3, LAYER.M3],
):
    """
    Connects together a pair of wg heaters and connect to a M3 port
    """

    cmp = Component()
    assert len(heater_ports) == 2
    assert (
        heater_ports[0].orientation == heater_ports[1].orientation
    ), "both ports should be facing in the same direction"
    angle = heater_ports[0].orientation
    angle = angle % 360
    assert angle in [0, 180], "angle should be 0 or 180, got {}".format(angle)

    dx = 0.0
    dy = 0.0

    angle_to_dps = {0: [(-dx, -dy), (-dx, dy)], 180: [(dx, -dy), (dx, dy)]}
    ports = heater_ports
    hw = heater_ports[0].width

    if angle in [0, 180]:
        ports.sort(key=lambda p: p.y)
    else:
        ports.sort(key=lambda p: p.x)

    _heater_to_metal = tlm(width=0.5, height=0.5, layers=tlm_layers, vias=[])

    tlm_positions = []
    for port, dp in zip(ports, angle_to_dps[angle]):
        # Extend heater
        p = port.midpoint

        # Add via/metal transitions
        tlm_pos = p + dp
        hm = _heater_to_metal.ref(position=tlm_pos)
        tlm_positions += [tlm_pos]
        cmp.add(hm)

    ss = 1 if angle == 0 else -1

    # Connect both sides with top metal
    edge_metal_piece_width = 7.0
    x = ss * edge_metal_piece_width / 2
    top_metal_layer = tlm_layers[-1]
    cmp.add_polygon(
        line(
            tlm_positions[0] + (x, -hw / 2),
            tlm_positions[1] + (x, hw / 2),
            edge_metal_piece_width,
        ),
        layer=top_metal_layer,
    )

    # Add metal port
    cmp.add_port(
        name="0",
        midpoint=0.5 * sum(tlm_positions) + (ss * edge_metal_piece_width / 2, 0),
        orientation=angle,
        width=metal_width,
        layer=top_metal_layer,
        port_type="dc",
    )

    return cmp


@deco_rename_ports
@autoname
def wg_heater_connected(
    waveguide_heater=waveguide_heater,
    wg_heater_connector=wg_heater_connector,
    tlm_layers=[LAYER.VIA1, LAYER.M1, LAYER.VIA2, LAYER.M2, LAYER.VIA3, LAYER.M3],
    **kwargs,
):
    """
    .. plot::
      :include-source:

      import pp

      c = pp.c.wg_heater_connected()
      pp.plotgds(c)

    """
    wg_heater = waveguide_heater(**kwargs)
    conn1 = wg_heater_connector(
        heater_ports=[wg_heater.ports["E0"], wg_heater.ports["E2"]],
        tlm_layers=tlm_layers,
    )

    conn2 = wg_heater_connector(
        heater_ports=[wg_heater.ports["W0"], wg_heater.ports["W2"]],
        tlm_layers=tlm_layers,
    )

    cmp = Component()
    for c in [wg_heater, conn1, conn2]:
        _c = cmp.add_ref(c)
        cmp.absorb(_c)

    for i, p in enumerate(wg_heater.get_optical_ports()):
        cmp.add_port(name=i, port=p)

    i += 1
    cmp.add_port(name=i, port=conn1.ports["0"])
    i += 1
    cmp.add_port(name=i, port=conn2.ports["0"])

    return cmp


def _demo_waveguide_heater():
    c = waveguide_heater(width=0.5)
    pp.write_gds(c)


if __name__ == "__main__":
    # print(c.get_optical_ports())
    c = wg_heater_connected(length=100.0, width=0.5)

    # c = waveguide_heater()
    # c = wg_heater_connector(heater_ports=[c.ports["W0"], c.ports["W1"]])
    pp.show(c)
