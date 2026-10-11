"""Port profiles must reproduce the taper's endpoint geometry when extended."""

import pytest

import gdsfactory as gf


@pytest.mark.parametrize(
    ("width1", "width2", "slab1", "slab2"), [(0.5, 0.5, 0.15, 6), (0.6, 0.8, 0.3, 8)]
)
def test_ridge_port_extensions(
    width1: float, width2: float, slab1: float, slab2: float
) -> None:
    taper = gf.c.taper_strip_to_ridge(
        width1=width1,
        width2=width2,
        w_slab1=slab1,
        w_slab2=slab2,
        layer_slab="SLAB150",
    )
    for name, width, slab in [("o1", width1, slab1), ("o2", width2, slab2)]:
        extension = gf.c.straight(length=5, cross_section=taper[name].cross_section)
        assert extension.area("WG") == pytest.approx(5 * width)
        assert extension.area("SLAB150") == pytest.approx(5 * slab)
        c = gf.Component()
        ref = c << extension
        ref.connect("o1", taper[name])
        assert ref["o1"].center == taper[name].center


@pytest.mark.parametrize(
    ("width", "slab_offset", "trench_width", "trench_offset"),
    [(0.5, 3, 2, 0.1), (0.8, 4, 1.5, 0.2)],
)
def test_trench_port_extensions(
    width: float, slab_offset: float, trench_width: float, trench_offset: float
) -> None:
    taper = gf.c.taper_strip_to_ridge_trenches(
        width=width,
        slab_offset=slab_offset,
        trench_width=trench_width,
        trench_offset=trench_offset,
        trench_layer="DEEP_ETCH",
    )
    for name, widening in [("o1", 0), ("o2", slab_offset)]:
        extension = gf.c.straight(length=5, cross_section=taper[name].cross_section)
        assert extension.area("WG") == pytest.approx(
            5 * (width + 2 * (trench_width - trench_offset + widening))
        )
        assert extension.area("DEEP_ETCH") == pytest.approx(
            10 * (trench_width + widening)
        )
        assert extension["o1"].width == width
        c = gf.Component()
        ref = c << extension
        ref.connect("o1", taper[name])
        assert ref["o1"].center == taper[name].center


@pytest.mark.parametrize("use_slab_port", [False, True])
def test_ridge_output_port_selection(use_slab_port: bool) -> None:
    taper = gf.c.taper_strip_to_ridge(use_slab_port=use_slab_port)
    port = taper["o2"]
    assert port.width == (6 if use_slab_port else 0.5)
    assert port.layer == gf.get_layer("SLAB90" if use_slab_port else "WG")
    extension = gf.c.straight(length=5, cross_section=port.cross_section)
    assert extension.area("WG") == pytest.approx(2.5)
    assert extension.area("SLAB90") == pytest.approx(30)


def test_ridge_slab_port_layer_override() -> None:
    taper = gf.c.taper_strip_to_ridge(slab_port_layer="SLAB90")
    assert len(taper.ports) == 2
    assert taper["o2"].layer == gf.get_layer("SLAB90")
    assert taper["o2"].width == 0.5
    extension = gf.c.straight(length=5, cross_section=taper["o2"].cross_section)
    assert extension.area("WG") == pytest.approx(2.5)
    assert extension.area("SLAB90") == pytest.approx(30)


@pytest.mark.parametrize(
    "component", ["taper_strip_to_ridge", "taper_strip_to_ridge_trenches"]
)
def test_taper_endpoint_frames(component: str) -> None:
    taper = gf.get_component(component, length=13)
    assert taper["o1"].center == (0, 0)
    assert taper["o1"].orientation == 180
    assert taper["o1"].mirror
    assert taper["o2"].center == (13, 0)
    assert taper["o2"].orientation == 0
    assert not taper["o2"].mirror
