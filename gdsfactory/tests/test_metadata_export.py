import toolz

import gdsfactory as gf


def test_metadata_export_partial():
    straight_wide = gf.partial(gf.components.straight, width=2)
    c = gf.components.mzi(straight=straight_wide)
    d = c.to_dict()
    assert d["settings"]["full"]["straight"]["width"] == 2
    assert d["settings"]["full"]["straight"]["function"] == "straight"


def test_metadata_export_function():
    c = gf.components.mzi()
    d = c.to_dict()
    assert d["settings"]["full"]["straight"]["function"] == "straight"


def test_metadata_export_compose():
    straight_wide = toolz.compose(gf.components.extend_ports, gf.components.straight)
    c = gf.components.mzi(straight=straight_wide)
    d = c.to_dict()
    assert d["settings"]["full"]["straight"][0]["function"] == "straight"
    assert d["settings"]["full"]["straight"][1]["function"] == "extend_ports"


if __name__ == "__main__":
    test_metadata_export_partial()
    test_metadata_export_function()
    test_metadata_export_compose()

    # c = gf.components.mzi()
    # d = c.to_dict()
    # print(d.settings.full.straight.function)

    # straight_wide = toolz.compose(gf.components.extend_ports, gf.components.straight)
    # c = gf.components.mzi(straight=straight_wide)
    # d = c.to_dict()

    # print(d.settings.full.straight)
    # df = d.settings.full
    # sf = df.straight
    # print(sf)
