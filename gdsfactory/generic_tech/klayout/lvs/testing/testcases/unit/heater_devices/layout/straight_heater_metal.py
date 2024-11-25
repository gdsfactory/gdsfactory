import os
from functools import partial

import gdsfactory as gf

straight_heater_metal_mk = (47, 1)
metal3 = gf.cross_section.metal3  # type: ignore[attr-defined]


@gf.cell
def straight_heater_metal_lvs() -> gf.Component:
    c = gf.Component()

    strip_hm10 = gf.cross_section.strip_heater_metal(heater_width=10)  # type: ignore[attr-defined]
    strip_hm35 = gf.cross_section.strip_heater_metal(heater_width=10)  # type: ignore[attr-defined]
    strip_hm100 = gf.cross_section.strip_heater_metal(heater_width=10)  # type: ignore[attr-defined]
    strip_hm50 = gf.cross_section.strip_heater_metal(heater_width=10)  # type: ignore[attr-defined]
    strip_hm30 = gf.cross_section.strip_heater_metal(heater_width=10)  # type: ignore[attr-defined]
    strip_hm40 = gf.cross_section.strip_heater_metal(heater_width=10)  # type: ignore[attr-defined]
    strip_hm500 = gf.cross_section.strip_heater_metal(heater_width=500)  # type: ignore[attr-defined]

    c1 = c << gf.components.straight_heater_metal(length=50, cross_section=strip_hm10)
    c1_mk = c << gf.components.rectangle(size=(50, 10), layer=straight_heater_metal_mk)

    c2 = c << gf.components.straight_heater_metal(length=70, cross_section=strip_hm35)
    c2_mk = c << gf.components.rectangle(size=(70, 35), layer=straight_heater_metal_mk)

    c3 = c << gf.components.straight_heater_metal(length=50, cross_section=strip_hm100)
    c3_mk = c << gf.components.rectangle(size=(50, 100), layer=straight_heater_metal_mk)

    c4 = c << gf.components.straight_heater_metal(length=100, cross_section=strip_hm50)
    c4_mk = c << gf.components.rectangle(size=(100, 50), layer=straight_heater_metal_mk)

    c5 = c << gf.components.straight_heater_metal(length=80, cross_section=strip_hm30)
    c5_mk = c << gf.components.rectangle(size=(80, 30), layer=straight_heater_metal_mk)

    c6 = c << gf.components.straight_heater_metal(length=60, cross_section=strip_hm40)
    c6_mk = c << gf.components.rectangle(size=(60, 40), layer=straight_heater_metal_mk)

    c7 = c << gf.components.straight_heater_metal(length=200, cross_section=strip_hm10)
    c7_mk = c << gf.components.rectangle(size=(200, 10), layer=straight_heater_metal_mk)

    c8 = c << gf.components.straight_heater_metal(length=120, cross_section=strip_hm100)
    c8_mk = c << gf.components.rectangle(
        size=(120, 100), layer=straight_heater_metal_mk
    )

    c9 = c << gf.components.straight_heater_metal(length=150, cross_section=strip_hm50)
    c9_mk = c << gf.components.rectangle(size=(150, 50), layer=straight_heater_metal_mk)

    c10 = c << gf.components.straight_heater_metal(
        length=500, cross_section=strip_hm500
    )
    c10_mk = c << gf.components.rectangle(
        size=(500, 500), layer=straight_heater_metal_mk
    )

    c1.dxmin = 0
    c2.dxmin = 0
    c3.dxmin = 0
    c4.dxmin = 0
    c5.dxmin = 0
    c6.dxmin = 0
    c7.dxmin = 0
    c8.dxmin = 0
    c9.dxmin = 0
    c10.dxmin = 0

    c1.dymin = 0
    c2.dymin = c1.dymax + 50
    c3.dymin = c2.dymax + 50
    c4.dymin = c3.dymax + 50
    c5.dymin = c4.dymax + 50
    c6.dymin = c5.dymax + 50
    c7.dymin = c6.dymax + 50
    c8.dymin = c7.dymax + 50
    c9.dymin = c8.dymax + 50
    c10.dymin = c9.dymax + 50

    c1_mk.dcenter = c1.dcenter
    c2_mk.dcenter = c2.dcenter
    c3_mk.dcenter = c3.dcenter
    c4_mk.dcenter = c4.dcenter
    c5_mk.dcenter = c5.dcenter
    c6_mk.dcenter = c6.dcenter
    c7_mk.dcenter = c7.dcenter
    c8_mk.dcenter = c8.dcenter
    c9_mk.dcenter = c9.dcenter
    c10_mk.dcenter = c10.dcenter

    route_single = partial(gf.routing.route_single, port_type="electrical")

    route_single(c, c1.ports["o1"], c2.ports["o1"], cross_section=metal3)
    route_single(c, c2.ports["o2"], c3.ports["o2"], cross_section=metal3)
    route_single(c, c3.ports["o1"], c4.ports["o1"], cross_section=metal3)
    route_single(c, c4.ports["o2"], c5.ports["o2"], cross_section=metal3)
    route_single(c, c5.ports["o1"], c6.ports["o1"], cross_section=metal3)
    route_single(c, c6.ports["o2"], c7.ports["o2"], cross_section=metal3)
    route_single(c, c7.ports["o1"], c8.ports["o1"], cross_section=metal3)
    route_single(c, c8.ports["o2"], c9.ports["o2"], cross_section=metal3)
    route_single(c, c9.ports["o1"], c10.ports["o1"], cross_section=metal3)
    return c


if __name__ == "__main__":
    testcase_path = os.path.dirname(os.path.abspath(__file__))
    heater_path = os.path.join(testcase_path, "straight_heater_metal.gds")

    c = straight_heater_metal_lvs()
    c.show()
    c.write_gds(heater_path)
