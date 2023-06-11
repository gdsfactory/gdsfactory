import os

import gdsfactory as gf

straight_heater_metal_mk = (47, 1)
m3_cross_section = gf.CrossSection(
    layer=(49, 0),
    width=10,
)


@gf.cell
def straight_heater_metal_lvs() -> gf.Component:
    c = gf.Component()

    c1 = c << gf.components.straight_heater_metal(length=50, heater_width=10)
    c1_mk = c << gf.components.rectangle(size=(50, 10), layer=straight_heater_metal_mk)

    c2 = c << gf.components.straight_heater_metal(length=70, heater_width=35)
    c2_mk = c << gf.components.rectangle(size=(70, 35), layer=straight_heater_metal_mk)

    c3 = c << gf.components.straight_heater_metal(length=50, heater_width=100)
    c3_mk = c << gf.components.rectangle(size=(50, 100), layer=straight_heater_metal_mk)

    c4 = c << gf.components.straight_heater_metal(length=100, heater_width=50)
    c4_mk = c << gf.components.rectangle(size=(100, 50), layer=straight_heater_metal_mk)

    c5 = c << gf.components.straight_heater_metal(length=80, heater_width=30)
    c5_mk = c << gf.components.rectangle(size=(80, 30), layer=straight_heater_metal_mk)

    c6 = c << gf.components.straight_heater_metal(length=60, heater_width=40)
    c6_mk = c << gf.components.rectangle(size=(60, 40), layer=straight_heater_metal_mk)

    c7 = c << gf.components.straight_heater_metal(length=200, heater_width=10)
    c7_mk = c << gf.components.rectangle(size=(200, 10), layer=straight_heater_metal_mk)

    c8 = c << gf.components.straight_heater_metal(length=120, heater_width=100)
    c8_mk = c << gf.components.rectangle(
        size=(120, 100), layer=straight_heater_metal_mk
    )

    c9 = c << gf.components.straight_heater_metal(length=150, heater_width=50)
    c9_mk = c << gf.components.rectangle(size=(150, 50), layer=straight_heater_metal_mk)

    c10 = c << gf.components.straight_heater_metal(length=500, heater_width=500)
    c10_mk = c << gf.components.rectangle(
        size=(500, 500), layer=straight_heater_metal_mk
    )

    c1.xmin = 0
    c2.xmin = 0
    c3.xmin = 0
    c4.xmin = 0
    c5.xmin = 0
    c6.xmin = 0
    c7.xmin = 0
    c8.xmin = 0
    c9.xmin = 0
    c10.xmin = 0

    c1.ymin = 0
    c2.ymin = c1.ymax + 50
    c3.ymin = c2.ymax + 50
    c4.ymin = c3.ymax + 50
    c5.ymin = c4.ymax + 50
    c6.ymin = c5.ymax + 50
    c7.ymin = c6.ymax + 50
    c8.ymin = c7.ymax + 50
    c9.ymin = c8.ymax + 50
    c10.ymin = c9.ymax + 50

    c1_mk.center = c1.center
    c2_mk.center = c2.center
    c3_mk.center = c3.center
    c4_mk.center = c4.center
    c5_mk.center = c5.center
    c6_mk.center = c6.center
    c7_mk.center = c7.center
    c8_mk.center = c8.center
    c9_mk.center = c9.center
    c10_mk.center = c10.center

    heater_12_route = gf.routing.get_route(
        c1.ports["o1"], c2.ports["o1"], cross_section=m3_cross_section
    )
    heater_23_route = gf.routing.get_route(
        c2.ports["o2"], c3.ports["o2"], cross_section=m3_cross_section
    )
    heater_34_route = gf.routing.get_route(
        c3.ports["o1"], c4.ports["o1"], cross_section=m3_cross_section
    )
    heater_45_route = gf.routing.get_route(
        c4.ports["o2"], c5.ports["o2"], cross_section=m3_cross_section
    )
    heater_56_route = gf.routing.get_route(
        c5.ports["o1"], c6.ports["o1"], cross_section=m3_cross_section
    )
    heater_67_route = gf.routing.get_route(
        c6.ports["o2"], c7.ports["o2"], cross_section=m3_cross_section
    )
    heater_78_route = gf.routing.get_route(
        c7.ports["o1"], c8.ports["o1"], cross_section=m3_cross_section
    )
    heater_89_route = gf.routing.get_route(
        c8.ports["o2"], c9.ports["o2"], cross_section=m3_cross_section
    )
    heater_910_route = gf.routing.get_route(
        c9.ports["o1"], c10.ports["o1"], cross_section=m3_cross_section
    )
    c.add(heater_12_route.references)
    c.add(heater_23_route.references)
    c.add(heater_34_route.references)
    c.add(heater_45_route.references)
    c.add(heater_56_route.references)
    c.add(heater_67_route.references)
    c.add(heater_78_route.references)
    c.add(heater_89_route.references)
    c.add(heater_910_route.references)
    return c


if __name__ == "__main__":
    testcase_path = os.path.dirname(os.path.abspath(__file__))
    heater_path = os.path.join(testcase_path, "straight_heater_metal.gds")

    c = straight_heater_metal_lvs()
    c.write_gds(heater_path)
