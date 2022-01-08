import gdsfactory as gf


def from_dphox(device: "dp.Device", foundry: "dp.foundry.Foundry") -> gf.Component:
    """Converts a Dphox Device into a gdsfactory Component.

    Note that you need to install dphox `pip install dphox`

    https://dphox.readthedocs.io/en/latest/index.html

    Args:
        device:
        foundry:
    """
    c = gf.Component(device.name)

    for layer_name, shapely_multipolygon in device.layer_to_polys.items():
        for poly in shapely_multipolygon:
            layer = foundry.layer_to_gds_label[layer_name]
            c.add_polygon(points=poly, layer=layer)

    for port_name, port in device.port.items():
        c.add_port(
            name=port_name,
            midpoint=(port.x, port.y),
            orientation=port.a,
            width=port.w,
        )
    return c


if __name__ == "__main__":
    import dphox as dp
    from dphox.demo import lateral_nems_ps

    nems_ps = lateral_nems_ps(waveguide_w=0.3)

    c = from_dphox(nems_ps, foundry=dp.foundry.FABLESS)
    c.show()
