import gdsfactory as gf


@gf.cell
def straight_with_pins(**kwargs):
    c = gf.Component()
    ref = c << gf.components.straight(**kwargs)
    c.add_ports(ref.ports)
    gf.add_pins.add_pins(c)
    return c


def test_metadata_export_pdk() -> None:
    c = gf.components.mzi(straight=straight_with_pins)
    assert c.settings.full["straight"]["function"] == "straight_with_pins"


if __name__ == "__main__":
    test_metadata_export_pdk()

    c = gf.components.mzi(straight=straight_with_pins)
    print(c.settings.full["straight"])
    c.show(show_ports=True)

    # df = d.settings.full
    # sf = df.straight
    # print(sf)
    # import inspect

    # func = straight_with_pins
    # sig = inspect.signature(func)
    # default = {p.name: p.default for p in sig.parameters.values()}
    # full = default.copy()
