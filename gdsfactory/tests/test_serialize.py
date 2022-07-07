import gdsfactory as gf
from gdsfactory.cross_section import strip


@gf.cell
def demo_cross_section_setting(cross_section=strip) -> gf.Component:
    return gf.components.straight(cross_section=cross_section)


def test_settings(data_regression, check: bool = True) -> None:
    """Avoid regressions when exporting settings."""
    component = demo_cross_section_setting()
    data_regression.check(component.to_dict())


if __name__ == "__main__":
    c = demo_cross_section_setting()
    d = c.to_dict()
    # c.show(show_ports=True)
    # test_settings()
