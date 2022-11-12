import gdsfactory as gf
from gdsfactory.cross_section import Section
from gdsfactory.difftest import difftest

WIDTH_WIDE = 2.0

xs_pin_m1 = gf.partial(
    gf.cross_section.strip_auto_widen,
    width=0.5,
    width_wide=WIDTH_WIDE,
    sections=(
        Section(width=1, offset=2, layer=gf.LAYER.NPP, name="n+"),
        Section(width=1, offset=3, layer=gf.LAYER.M1, name="m1"),
    ),
)

xs_pin = gf.partial(
    gf.cross_section.strip_auto_widen,
    sections=(Section(width=1, offset=2, layer=gf.LAYER.NPP, name="n+"),),
)


@gf.cell
def taper_pin(length: float = 5, **kwargs) -> gf.Component:
    trans = gf.path.transition(
        cross_section1=xs_pin(),
        cross_section2=xs_pin(width=WIDTH_WIDE),
        width_type="linear",
    )
    path = gf.path.straight(length=length)
    return gf.path.extrude(path, cross_section=trans)


def test_get_route_auto_widen() -> gf.Component:
    c = gf.Component("test_get_route_auto_widen")
    route = gf.routing.get_route_from_waypoints(
        [(0, 0), (300, 0), (300, 300), (-600, 300), (-600, -300)],
        cross_section=xs_pin_m1,
        bend=gf.partial(gf.components.bend_euler, cross_section=xs_pin),
        taper=taper_pin,
        radius=30,
    )
    c.add(route.references)
    difftest(c)
    return c


if __name__ == "__main__":
    c = gf.Component()
    route = gf.routing.get_route_from_waypoints(
        # [(0, 0), (300, 0), (300, 300), (-600, 300), (-600, -300)],
        [(0, 0), (300, 0), (300, 300), (300, 600), (600, 600)],
        cross_section=xs_pin_m1,
        bend=gf.partial(gf.components.bend_euler, cross_section=xs_pin),
        taper=taper_pin,
        radius=30,
    )
    c.add(route.references)
    c.show(show_ports=True)
