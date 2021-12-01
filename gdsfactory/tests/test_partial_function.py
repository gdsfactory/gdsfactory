import gdsfactory as gf


def test_partial_function_with_kwargs():
    mmi400 = gf.partial(gf.components.mmi1x2, width=0.4)
    mmi400_args = gf.partial(gf.components.mmi1x2, 0.4)
    mmi600 = gf.partial(gf.components.mmi1x2, width=0.6)
    mzi400 = gf.partial(gf.components.mzi, splitter=mmi400)
    mzi600 = gf.partial(gf.components.mzi, splitter=mmi600)

    c400 = mzi400()
    c600 = mzi600()

    assert c600.name != c400.name, f"{c600.name} must be different from {c400.name}"

    cmmi400 = mmi400()
    cmmi400_args = mmi400_args()
    assert (
        cmmi400_args.name == cmmi400.name
    ), f"{cmmi400_args.name} must be equal to {cmmi400.name}"


def test_partial_function_without_kwargs():
    r1 = gf.partial(gf.c.rectangle, size=(4, 2))
    r2 = gf.partial(gf.c.rectangle, size=(4, 2))
    r3 = gf.partial(gf.c.rectangle, (4, 2))

    c1 = r1()
    c2 = r2()
    c3 = r3()

    assert c1.name == c2.name == c3.name, f"{c1.name} == {c2.name} == {c3.name}"


if __name__ == "__main__":
    # test_partial_function_with_kwargs()
    test_partial_function_without_kwargs()
    # mmi400 = gf.partial(gf.components.mmi1x2, width=0.4)
    # mmi600 = gf.partial(gf.components.mmi1x2, width=0.6)
    # mzi400 = gf.partial(gf.components.mzi, splitter=mmi400)
    # mzi600 = gf.partial(gf.components.mzi, splitter=mmi600)
    # c400 = mzi400()
    # c600 = mzi600()
    # print(c400.name, c600.name)
