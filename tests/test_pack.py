import gdsfactory as gf


def test_pack_same_name():
    p = gf.pack(
        [gf.components.straight(length=i) for i in [1, 1]],
        spacing=20.0,
        max_size=(100, 100),
        text_prefix="R",
        name_prefix="demo",
        text_anchors=["nc"],
        text_offsets=[(-10, 0)],
        text_mirror=True,
        v_mirror=True,
    )
    c = p[0]
    assert c
