"""write xsection script for KLayout plugin.

https://gdsfactory.github.io/klayout_pyxs/DocGrow.html

"""

from __future__ import annotations

import pathlib

from gdsfactory.generic_tech import LAYER  # type: ignore

nm = 1e-3


def layer_to_string(layer: tuple[int, int]) -> str:
    return f"{layer[0]}/{layer[1]}"


def get_klayout_pyxs(
    t_box: float = 1.0,
    t_slab: float = 90 * nm,
    t_si: float = 0.22,
    t_ge: float = 0.4,
    t_nitride: float = 0.4,
    h_etch1: float = 0.07,
    h_etch2: float = 0.06,
    h_etch3: float = 0.09,
    t_clad: float = 0.6,
    t_m1: float = 0.5,
    t_m2: float = 0.5,
    t_m3: float = 2.0,
    gap_m1_m2: float = 0.6,
    gap_m2_m3: float = 0.3,
    t_heater: float = 0.1,
    gap_oxide_nitride: float = 0.82,
    t_m1_oxide: float = 0.6,
    t_m2_oxide: float = 2.0,
    t_m3_oxide: float = 0.5,
    layer_wg: tuple[int, int] = LAYER.WG,
    layer_fc: tuple[int, int] = LAYER.SLAB150,
    layer_rib: tuple[int, int] = LAYER.SLAB90,
    layer_n: tuple[int, int] = LAYER.N,
    layer_np: tuple[int, int] = LAYER.NP,
    layer_npp: tuple[int, int] = LAYER.NPP,
    layer_p: tuple[int, int] = LAYER.P,
    layer_pp: tuple[int, int] = LAYER.PP,
    layer_ppp: tuple[int, int] = LAYER.PPP,
    layer_PDPP: tuple[int, int] = LAYER.GEP,
    layer_nitride: tuple[int, int] = LAYER.WGN,
    layer_Ge: tuple[int, int] = LAYER.GE,
    layer_GePPp: tuple[int, int] = LAYER.GEP,
    layer_GeNPP: tuple[int, int] = LAYER.GEN,
    layer_viac: tuple[int, int] = LAYER.VIAC,
    layer_viac_slot: tuple[int, int] = LAYER.VIAC,
    layer_m1: tuple[int, int] = LAYER.M1,
    layer_mh: tuple[int, int] = LAYER.HEATER,
    layer_via1: tuple[int, int] = LAYER.VIA1,
    layer_m2: tuple[int, int] = LAYER.M2,
    layer_via2: tuple[int, int] = LAYER.VIA2,
    layer_m3: tuple[int, int] = LAYER.M3,
    layer_open: tuple[int, int] = LAYER.PADOPEN,
) -> str:
    """Returns klayout_pyxs plugin script to show chip cross-section in klayout.

    https://gdsfactory.github.io/klayout_pyxs/DocGrow.html

    """
    return f"""

t_box={t_box}
t_slab={t_slab}
t_si={t_si}
t_ge={t_ge}
t_nitride={t_nitride}
h_etch1={h_etch1}
h_etch2={h_etch2}
h_etch3={h_etch3}
t_clad={t_clad}
t_m1={t_m1}
t_m2={t_m2}
t_m3={t_m3}
t_heater={t_heater}
gap_m1_m2={gap_m1_m2}
gap_m2_m3={gap_m2_m3}
gap_oxide_nitride={gap_oxide_nitride}
t_m1_oxide={t_m1_oxide}
t_m2_oxide={t_m2_oxide}
t_m3_oxide={t_m3_oxide}

l_wg = layer({layer_to_string(layer_wg)!r})
l_fc = layer({layer_to_string(layer_fc)!r})
l_rib = layer({layer_to_string(layer_rib)!r})

l_n = layer({layer_to_string(layer_n)!r})
l_np = layer({layer_to_string(layer_np)!r})
l_npp = layer({layer_to_string(layer_npp)!r})
l_p = layer({layer_to_string(layer_p)!r})
l_pp = layer({layer_to_string(layer_pp)!r})
l_ppp = layer({layer_to_string(layer_ppp)!r})
l_PDPP = layer({layer_to_string(layer_PDPP)!r})
l_bottom_implant = l_PDPP

l_nitride = layer({layer_to_string(layer_nitride)!r})
l_Ge = layer({layer_to_string(layer_Ge)!r})
l_GePPp = layer({layer_to_string(layer_GePPp)!r})
l_GeNPP = layer({layer_to_string(layer_GeNPP)!r})

l_viac = layer({layer_to_string(layer_viac)!r})
l_viac_slot = layer({layer_to_string(layer_viac_slot)!r})
l_m1 = layer({layer_to_string(layer_m1)!r})
l_mh = layer({layer_to_string(layer_mh)!r})
l_via1 = layer({layer_to_string(layer_via1)!r})
l_m2 = layer({layer_to_string(layer_m2)!r})
l_via2 = layer({layer_to_string(layer_via2)!r})
l_m3 = layer({layer_to_string(layer_m3)!r})
l_open = layer({layer_to_string(layer_open)!r})

l_top_implant = l_GePPp.or_(l_GeNPP)
l_viac = l_viac.or_(l_viac_slot)

# Declare the basic accuracy used to remove artifacts for example: delta(5 * dbu)
delta(dbu)
depth(12.0)
height(12.0)

################ front-end

l_wg_etch1 = l_wg.inverted()  # protects ridge
l_wg_etch2 = (
    l_fc.or_(l_wg)
).inverted()  # protects ridge and grating couplers from the etch down to the slab (forms rib straights)
l_wg_etch3 = (
    l_rib.or_(l_fc).or_(l_wg)
).inverted()  # protects ridge, grating couplers and rib straights from the final etch to form strip straights


################ back-end
substrate = bulk
box = deposit(t_box)
si = deposit(t_si)

################ silicon etch to for the passives
mask(l_wg_etch1).etch(
    h_etch1, 0.0, mode="round", into=[si]
)  # 70nm etch for GC, rib and strip
mask(l_wg_etch2).etch(
    h_etch2, 0.0, mode="round", into=[si]
)  # 60nm etch after 70nm = 130nm etch (90nm slab)
mask(l_wg_etch3).etch(
    h_etch3, 0.0, mode="round", into=[si]
)  # etches the remaining 90nm slab for strip straights

output("300/0", box)
output("301/0", si)

############### doping
mask(l_bottom_implant).etch(t_si, 0.0, mode="round", into=[si])
bottom_implant = mask(l_bottom_implant).grow(t_si, 0.0, mode="round")

mask(l_n).etch(t_slab, 0.0, mode="round", into=[si])
n = mask(l_n).grow(t_slab, 0.0, mode="round")

mask(l_p).etch(t_slab, 0.0, mode="round", into=[si])
p = mask(l_p).grow(t_slab, 0.0, mode="round")

mask(l_np).etch(t_slab, 0.0, mode="round", into=[n, p, si, bottom_implant])
np = mask(l_np).grow(t_slab, 0.0, mode="round")

mask(l_pp).etch(t_slab, 0.0, mode="round", into=[n, p, si, bottom_implant])
pp = mask(l_pp).grow(t_slab, 0.0, mode="round")

mask(l_npp).etch(t_slab, 0.0, mode="round", into=[n, p, np, pp, si, bottom_implant])
npp = mask(l_npp).grow(t_slab, 0.0, mode="round")

mask(l_ppp).etch(t_slab, 0.0, mode="round", into=[n, p, np, pp, si, bottom_implant])
ppp = mask(l_ppp).grow(t_slab, 0.0, mode="round")

output("327/0", bottom_implant)
output("330/0", p)
output("320/0", n)
output("321/0", npp)
output("331/0", ppp)

################ Ge
Ge = mask(l_Ge).grow(t_ge, 0, bias=0.0, taper=10)
output("315/0", Ge)

################ Nitride
ox_nitride = deposit(2 * gap_oxide_nitride, 2 * gap_oxide_nitride)
planarize(less=gap_oxide_nitride, into=[ox_nitride])
output("302/0", ox_nitride)

nitride = mask(l_nitride).grow(t_nitride, 0, bias=0.0, taper=10)
output("305/0", nitride)

################# back-end
################# VIAC, M1 and MH
ox_nitride_clad = deposit(t_clad + t_ge + t_nitride, t_clad + t_ge + t_nitride, mode="round")

planarize(less=t_ge + t_nitride, into=[ox_nitride_clad])
mask(l_viac).etch(
    t_clad + t_ge + t_nitride + gap_oxide_nitride, taper=4, into=[ox_nitride_clad, ox_nitride]
)

viac = deposit(2 * t_clad, 2 * t_clad)
planarize(less=2 * t_clad, into=[viac])

mh = deposit(t_heater, t_heater)
mask(l_mh.inverted()).etch(t_heater + t_heater, into=[mh])
m1 = deposit(t_m1, t_m1)
mask(l_m1.inverted()).etch(t_m1 + t_m1, into=[m1])
output("306/0", mh)
output("399/0", m1)

output("304/0", ox_nitride_clad)
output("303/0", viac)

################# VIA1 and M2
ox_m1 = deposit(2 * t_m1_oxide, 2 * t_m1_oxide, mode="round")
planarize(less=t_m1_oxide, into=[ox_m1])

mask(l_via1).etch(t_m1_oxide + gap_m1_m2, taper=4, into=[ox_m1])
via1 = deposit(t_m2, t_m2)

mask(l_m2.inverted()).etch(t_m2, taper=4, into=[via1])
output("308/0", via1)

ox_m2 = deposit(2 * t_m2_oxide, 2 * t_m2_oxide, mode="round")
planarize(less=t_m2_oxide, into=[ox_m2])
output("309/0", ox_m2)
output("307/0", ox_m1)

################# VIA2 and M3
mask(l_via2).etch(t_m2_oxide + gap_m2_m3, taper=4, into=[ox_m2, ox_m2])
via2 = deposit(t_m3, t_m3)
mask(l_m3.inverted()).etch(t_m3, taper=4, into=[via2])
output("310/0", via2)

################# passivation and ML Open
ox_m3 = deposit(t_m3_oxide, t_m3_oxide, mode="round")
mask(l_open).etch(t_m3_oxide + t_m3_oxide, into=[ox_m3], taper=5)
output("311/0", ox_m3)
"""


if __name__ == "__main__":
    script = get_klayout_pyxs(
        t_box=2.0,
        t_slab=110 * nm,
        t_si=220 * nm,
        t_ge=400 * nm,
        t_nitride=400 * nm,
        h_etch1=0.07,
        h_etch2=0.06,
        h_etch3=0.09,
        t_clad=0.6,
        t_m1=0.5,
        t_m2=0.5,
        t_m3=2.0,
        gap_m1_m2=0.6,
        gap_m2_m3=0.3,
        t_heater=0.1,
        gap_oxide_nitride=0.82,
        t_m1_oxide=0.6,
        t_m2_oxide=2.0,
        t_m3_oxide=0.5,
        layer_wg=LAYER.WG,
        layer_fc=LAYER.SLAB150,
        layer_rib=LAYER.SLAB90,
        layer_n=LAYER.N,
        layer_np=LAYER.NP,
        layer_npp=LAYER.NPP,
        layer_p=LAYER.P,
        layer_pp=LAYER.PP,
        layer_ppp=LAYER.PPP,
        layer_PDPP=LAYER.GEP,
        layer_nitride=LAYER.WGN,
        layer_Ge=LAYER.GE,
        layer_GePPp=LAYER.GEP,
        layer_GeNPP=LAYER.GEN,
        layer_viac=LAYER.VIAC,
        layer_viac_slot=LAYER.VIAC,
        layer_m1=LAYER.M1,
        layer_mh=LAYER.HEATER,
        layer_via1=LAYER.VIA1,
        layer_m2=LAYER.M2,
        layer_via2=LAYER.VIA2,
        layer_m3=LAYER.M3,
        layer_open=LAYER.PADOPEN,
    )

    script_path = (
        pathlib.Path(__file__).parent.absolute()
        / "klayout"
        / "tech"
        / "xsection_planarized.pyxs"
    )
    print(script_path)
    script_path.write_text(script)
    print(script)
