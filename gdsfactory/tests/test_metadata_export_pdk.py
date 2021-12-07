import gdsfactory as gf


@gf.cell
def straight_with_pins(**kwargs):
    c = gf.Component()
    ref = c << gf.c.straight()
    c.add_ports(ref.ports)
    gf.add_pins(c)
    return c


def test_metadata_export_pdk():
    c = gf.c.mzi(straight=straight_with_pins)
    d = c.to_dict_config()
    assert d.info.full.straight.function == "straight_with_pins"


if __name__ == "__main__":
    test_metadata_export_pdk()

    c = gf.c.mzi(straight=straight_with_pins)
    d = c.to_dict_config()
    print(d.info.full.straight)

    # df = d.info.full
    # sf = df.straight
    # print(sf)

    import inspect

    func = straight_with_pins
    sig = inspect.signature(func)
    default = {p.name: p.default for p in sig.parameters.values()}
    full = default.copy()
