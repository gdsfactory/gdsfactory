"""
CMOS side-gate transistors from
F. Zanetto et al., “Unconventional Monolithic Electronics in a
Conventional Silicon Photonics Platform,” IEEE Trans. Electron Devices,
vol. 70, no. 10, pp. 4993-4998, Oct. 2023, doi: 10.1109/TED.2023.3304268.
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component

# TODO: find correct imports


@cell
def cmos_side_gate(
    is_nmos: bool = True,
) -> Component:
    """
    Returns a CMOS side-gate transistor.

    Args:
        is_nmos: True if NMOS, False if PMOS (determines doping)

    Notes:
        TODO
    """

    ### Questions for standup
    #   Need to do anything for 'native' silicon layer? Oxide layer?
    #       Need to draw WG/core (native silicon)
    #       Oxide layer there by default
    #   How to do layered doping?
    #       Draw on top of each other
    #       strongly-doped layer is naturally thinner
    #   Need to do 3D?
    #       Note: need to add_polygon from LayerSpec to do 3D
    #   How to do vias properly?
    #       Use via stack components in gdsfactory
    #       terminate cell on metal layer (M1, M2)
    #       Size of vias?
    #           Foundry has rules on enclosure
    #           Need to maintain gap between edge of via and edge of body, add as parameter to cell
    #               Use a good default (i.e. approximate from figure)
    #   How to do metal layers? -- draw in M1 (or use routing function, overkill)
    #       Expose ports -- look in documentation when layout finished
    #       Use well-labeled ports (i.e. source, drain, gate)
    #   pMOS: flipping implants (n-->p, p-->n)
    #       Better implementation: create single MOSFET, implement nMOS and pMOS
    #       as partial functions with different defaults

    t_ox = 0.2  # 200 nm
    # W = 0.22  # 220 nm
    # L = 4  # 4 um

    # represents strip of weak doping between gate and oxide layer
    gate_epsilon = 0.4  # 400 nm, guess based on figure

    dope_gate_strong = "PPP"
    dope_gate_weak = "PP"
    dope_bulk_strong = "PPP"
    dope_bulk_weak = "PP"
    dope_source_drain_strong = "NPP"
    dope_source_drain_weak = "NP"

    if not is_nmos:
        dope_gate_strong = "NPP"
        dope_gate_weak = "NP"
        dope_bulk_strong = "NPP"
        dope_bulk_weak = "NP"
        dope_source_drain_strong = "PPP"
        dope_source_drain_weak = "PP"

    c = Component("cmos_side_gate")

    native = gf.components.rectangle(size=(6, 16), layer="WG", centered=True)
    native_gate1 = gf.components.rectangle(size=(4, 4), layer="WG", centered=True)
    native_gate2 = gf.components.rectangle(size=(4, 4), layer="WG", centered=True)
    _ = c << native
    ref_native_gate1 = c << native_gate1
    ref_native_gate2 = c << native_gate2
    ref_native_gate1.movex(-5)
    ref_native_gate1.movex(-t_ox)
    ref_native_gate2.movex(5)
    ref_native_gate2.movex(t_ox)

    gate1 = gf.components.rectangle(
        size=(4 - gate_epsilon, 4), layer=dope_gate_weak, centered=True
    )
    gate2 = gf.components.rectangle(
        size=(4 - gate_epsilon, 4), layer=dope_gate_weak, centered=True
    )
    ref_gate1 = c << gate1
    ref_gate2 = c << gate2
    ref_gate1.movex(-5 - t_ox - gate_epsilon / 2)
    ref_gate2.movex(5 + t_ox + gate_epsilon / 2)

    gate1_strong = gf.components.rectangle(
        size=(4, 4), layer=dope_gate_strong, centered=True
    )
    gate2_strong = gf.components.rectangle(
        size=(4, 4), layer=dope_gate_strong, centered=True
    )
    ref_gate1_strong = c << gate1_strong
    ref_gate2_strong = c << gate2_strong
    ref_gate1_strong.movex(-5 - t_ox)
    ref_gate2_strong.movex(5 + t_ox)

    bulk = gf.components.rectangle(size=(2, 12), layer=dope_bulk_weak, centered=True)
    ref_bulk = c << bulk
    ref_bulk.movey(-2)

    bulk_strong = gf.components.rectangle(
        size=(2, 3), layer=dope_bulk_strong, centered=True
    )
    ref_bulk_strong = c << bulk_strong
    ref_bulk_strong.movey(-6.5)

    source1 = gf.components.rectangle(
        size=(2, 3), layer=dope_source_drain_weak, centered=True
    )
    source2 = gf.components.rectangle(
        size=(2, 3), layer=dope_source_drain_weak, centered=True
    )
    ref_source1 = c << source1
    ref_source2 = c << source2
    ref_source1.movey(-6.5)
    ref_source2.movey(-6.5)
    ref_source1.movex(-2)
    ref_source2.movex(2)

    source1_strong = gf.components.rectangle(
        size=(2, 3), layer=dope_source_drain_strong, centered=True
    )
    source2_strong = gf.components.rectangle(
        size=(2, 3), layer=dope_source_drain_strong, centered=True
    )
    ref_source1_strong = c << source1_strong
    ref_source2_strong = c << source2_strong
    ref_source1_strong.movey(-6.5)
    ref_source2_strong.movey(-6.5)
    ref_source1_strong.movex(-2)
    ref_source2_strong.movex(2)

    source_wing1 = gf.components.rectangle(
        size=(1, 3), layer=dope_source_drain_weak, centered=True
    )
    source_wing2 = gf.components.rectangle(
        size=(1, 3), layer=dope_source_drain_weak, centered=True
    )
    ref_source_wing1 = c << source_wing1
    ref_source_wing2 = c << source_wing2
    ref_source_wing1.movey(-3.5)
    ref_source_wing2.movey(-3.5)
    ref_source_wing1.movex(-2.5)
    ref_source_wing2.movex(2.5)

    drain = gf.components.rectangle(
        size=(6, 3), layer=dope_source_drain_weak, centered=True
    )
    ref_drain = c << drain
    ref_drain.movey(6.5)

    drain_strong = gf.components.rectangle(
        size=(6, 3), layer=dope_source_drain_strong, centered=True
    )
    ref_drain_strong = c << drain_strong
    ref_drain_strong.movey(6.5)

    drain_wing1 = gf.components.rectangle(
        size=(1, 3), layer=dope_source_drain_weak, centered=True
    )
    drain_wing2 = gf.components.rectangle(
        size=(1, 3), layer=dope_source_drain_weak, centered=True
    )
    ref_drain_wing1 = c << drain_wing1
    ref_drain_wing2 = c << drain_wing2
    ref_drain_wing1.movey(3.5)
    ref_drain_wing2.movey(3.5)
    ref_drain_wing1.movex(-2.5)
    ref_drain_wing2.movex(2.5)

    # vias
    via_gate1 = gf.components.via(size=(2, 2))
    via_gate2 = gf.components.via(size=(2, 2))
    ref_via_gate1 = c << via_gate1
    ref_via_gate2 = c << via_gate2
    ref_via_gate1.movex(-5 - t_ox)
    ref_via_gate2.movex(5 + t_ox)

    via_source = gf.components.via(size=(4, 2))
    via_drain = gf.components.via(size=(4, 2))
    ref_via_source = c << via_source
    ref_via_drain = c << via_drain
    ref_via_source.movey(-6.5)
    ref_via_drain.movey(6.5)

    # metal
    metal_gate = gf.components.rectangle(
        size=(14, 4), layer="M1", port_type="electrical", centered=True
    )
    ref_metal_gate = c << metal_gate

    metal_source = gf.components.rectangle(
        size=(4, 2), layer="M1", port_type="electrical", centered=True
    )
    metal_drain = gf.components.rectangle(
        size=(4, 2), layer="M1", port_type="electrical", centered=True
    )
    ref_metal_source = c << metal_source
    ref_metal_drain = c << metal_drain
    ref_metal_source.movey(-6.5)
    ref_metal_drain.movey(6.5)

    gate_all_ports = ref_metal_gate.get_ports_list()
    gate_ports = [gate_all_ports[0], gate_all_ports[2]]
    drain_all_ports = ref_metal_drain.get_ports_list()
    drain_ports = [drain_all_ports[0], drain_all_ports[1], drain_all_ports[2]]
    source_all_ports = ref_metal_source.get_ports_list()
    source_ports = [source_all_ports[0], source_all_ports[2], source_all_ports[3]]

    # connections to the metal layers
    c.add_ports(source_ports, prefix="source_")
    c.add_ports(drain_ports, prefix="drain_")
    c.add_ports(gate_ports, prefix="gate_")

    return c


if __name__ == "__main__":
    pass
    c1 = cmos_side_gate(is_nmos=True)
    c1.show(show_ports=True)

    c2 = cmos_side_gate(is_nmos=False)
    c2.show(show_ports=True)
