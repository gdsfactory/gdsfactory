import gdsfactory as gf


def test_name_partial_functions():
    s1 = gf.partial(gf.c.straight)
    s2 = gf.partial(gf.c.straight, length=5)
    s3 = gf.partial(gf.c.straight, 5)

    m1 = gf.partial(gf.c.mzi, straight=s1)()
    m2 = gf.partial(gf.c.mzi, straight=s2)()
    m3 = gf.partial(gf.c.mzi, straight=s3)()

    # print(m1.name)
    # print(m2.name)
    # print(m3.name)

    assert (
        m2.name == m3.name
    ), f"{m2.name} different from {m2.name} while they are the same function"
    assert (
        m1.name != m2.name
    ), f"{m1.name} is the same {m2.name} while they are different functions"
    assert (
        m1.name != m3.name
    ), f"{m1.name} is the same {m3.name} while they are different functions"


if __name__ == "__main__":
    test_name_partial_functions()
