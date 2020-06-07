from phidl.device_layout import _parse_layer
from pp.geo_utils import polygon_grow


def add_keepout(c, target_layers, keepout_layers, margin=2.0):
    """
    Lookup all polygons in this cell. You can also use add_padding
    """
    all_cells = list(c.get_dependencies(True)) + [c]
    for _c in all_cells:
        for _layer in target_layers:
            polys_by_layer = _c.get_polygons(by_spec=True)
            layer = _parse_layer(_layer)
            for ko_layer in keepout_layers:
                ko_layer = _parse_layer(ko_layer)
                polys = polys_by_layer[layer] if layer in polys_by_layer else []
                if not polys:
                    continue

                ko_polys_pts = [polygon_grow(poly, margin) for poly in polys]
                _c.add_polygon(ko_polys_pts, ko_layer)


if __name__ == "__main__":
    from pp.components.crossing_waveguide import crossing_etched
    from pp.components.crossing_waveguide import crossing45
    from pp.layers import LAYER

    c = crossing45(alpha=0.5, crossing=crossing_etched)

    target_layers = [LAYER.WG, LAYER.SLAB150]
    keepout_layers = [LAYER.NO_TILE_SI]

    add_keepout(c, target_layers, keepout_layers)
    import pp

    pp.show(c)
