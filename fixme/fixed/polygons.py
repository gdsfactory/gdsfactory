import gdsfactory as gf


@gf.cell
def wrap_polygon(polygon) -> gf.Component:
    c = gf.Component()
    # c.add_polygon(polygon, layer="WG")
    return c


@gf.cell
def wrap_polygons(polygons) -> gf.Component:
    c = gf.Component()
    # c.add_polygon(polygons, layer="WG")
    return c


if __name__ == "__main__":
    import gdstk

    # Try individually toggling on/off the commented lines below to see failing/passing cases

    c = wrap_polygon(gdstk.rectangle((0, 0), (1, 1)))  # FAILS
    s = gf.components.straight()

    # c = wrap_polygons(s.get_polygons(as_array=False))  # FAILS
    # c = wrap_polygons(s.get_polygons(by_spec=False, as_array=True))  # WORKS
    # c = wrap_polygons(s.get_polygons(by_spec=True, as_array=True))  # WORKS

    s = gf.components.ring_double_heater()
    c = wrap_polygons(s.get_polygons(by_spec=False, as_array=True))  # FAILS
    # c = wrap_polygons(s.get_polygons(by_spec=(1, 0), as_array=True))  # FAILS
    # c = wrap_polygons(s.get_polygons(by_spec=True, as_array=True))  # FAILS
    # c.show()

    import numpy as np

    p = s.get_polygons(by_spec=False, as_array=True)  # FAILS
    # value = np.round(p, 8)
    # d = orjson.loads(orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY))
    value = np.round(p, 8)
