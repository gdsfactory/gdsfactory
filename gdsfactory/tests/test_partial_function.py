import gdsfactory as gf


def test_partial_function():
    mmi400 = gf.partial(gf.components.mmi1x2, width=0.4)
    mmi600 = gf.partial(gf.components.mmi1x2, width=0.6)
    mzi400 = gf.partial(gf.components.mzi, splitter=mmi400)
    mzi600 = gf.partial(gf.components.mzi, splitter=mmi600)
    c400 = mzi400()
    c600 = mzi600()
    assert c600.name != c400.name, print(c400.name, c600.name)


if __name__ == "__main__":
    mmi400 = gf.partial(gf.components.mmi1x2, width=0.4)
    mmi600 = gf.partial(gf.components.mmi1x2, width=0.6)
    mzi400 = gf.partial(gf.components.mzi, splitter=mmi400)
    mzi600 = gf.partial(gf.components.mzi, splitter=mmi600)
    c400 = mzi400()
    c600 = mzi600()
    print(c400.name, c600.name)
