import gdsfactory as gf


def test_coupler_full() -> None:
    gf.components.coupler_full(cross_section="strip")
    gf.components.coupler_full(cross_section="strip", width=10)
