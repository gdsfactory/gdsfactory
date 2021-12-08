import toolz

import gdsfactory as gf


def test_metadata_export_partial():
    straight_wide = gf.partial(gf.c.straight, width=2)
    c = gf.c.mzi(straight=straight_wide)
    d = c.to_dict_config()
    assert d.info.full.straight.width == 2
    assert d.info.full.straight.function == "straight"


def test_metadata_export_function():
    gf.clear_cache()
    c = gf.c.mzi()
    d = c.to_dict_config()
    assert d.info.full.straight.function == "straight"


def test_metadata_export_compose():
    straight_wide = toolz.compose(gf.c.extend_ports, gf.c.straight)
    c = gf.c.mzi(straight=straight_wide)
    d = c.to_dict_config()
    assert d.info.full.straight[0]["function"] == "straight"
    assert d.info.full.straight[1]["function"] == "extend_ports"


if __name__ == "__main__":
    test_metadata_export_compose()

    # test_metadata_export_function()
    # c = gf.c.mzi()
    # d = c.to_dict_config()
    # print(d.info.full.straight.function)

    # test_metadata_export_partial()
    # from gdsfactory.cell import clean_name
    # from gdsfactory.component import clean_dict
    # import inspect

    # straight_wide = gf.partial(gf.c.straight, width=2)
    # func = gf.c.mzi
    # func = gf.partial(func, straight=straight_wide)
    # c = gf.c.mzi(straight=straight_wide)
    # sig = inspect.signature(func)
    # default = {p.name: p.default for p in sig.parameters.values()}
    # full = default.copy()
    # clean_dict(full)

    # straight_wide = gf.partial(gf.c.straight, width=2)

    straight_wide = toolz.compose(gf.c.extend_ports, gf.c.straight)
    c = gf.c.mzi(straight=straight_wide)
    d = c.to_dict_config()
    print(d.info.full.straight)

    # df = d.info.full
    # sf = df.straight
    # print(sf)
