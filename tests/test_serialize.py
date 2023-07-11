from __future__ import annotations

import gdstk

import gdsfactory as gf
from gdsfactory.cross_section import strip


@gf.cell
def demo_cross_section_setting(cross_section=strip) -> gf.Component:
    return gf.components.straight(cross_section=cross_section)


def test_settings(data_regression, check: bool = True) -> None:
    """Avoid regressions when exporting settings."""
    component = demo_cross_section_setting()
    settings = component.to_dict()
    if data_regression:
        data_regression.check(settings)


@gf.cell
def wrap_polygon(polygon) -> gf.Component:
    return gf.Component()


@gf.cell
def wrap_polygons(polygons) -> gf.Component:
    return gf.Component()


def test_serialize_polygons() -> None:
    wrap_polygon(gdstk.rectangle((0, 0), (1, 1)))  # FAILS

    s = gf.components.straight()
    wrap_polygons(s.get_polygons(as_array=False))  # FAILS
    wrap_polygons(s.get_polygons(by_spec=False, as_array=True))  # WORKS
    wrap_polygons(s.get_polygons(by_spec=True, as_array=True))  # WORKS

    s = gf.components.ring_double_heater()
    wrap_polygons(s.get_polygons(by_spec=False, as_array=True))  # FAILS
    wrap_polygons(s.get_polygons(by_spec=(1, 0), as_array=True))  # FAILS
    wrap_polygons(s.get_polygons(by_spec=True, as_array=True))  # FAILS


if __name__ == "__main__":
    c1 = gf.components.mmi1x2()
    settings = c1.settings.full
    cell_name = c1.settings.function_name
    c2 = gf.get_component({"component": cell_name, "settings": settings})
    # c = demo_cross_section_setting()
    # d = c.to_dict()
    # c.show(show_ports=True)
    # test_settings(None)
    # test_serialize_polygons()
