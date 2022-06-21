import gdsfactory as gf


def test_add_settings_label():
    c = gf.c.mzi(delta_length=20, decorator=gf.functions.add_settings_label)
    assert c


if __name__ == "__main__":
    c = gf.c.mzi(delta_length=20, decorator=gf.functions.add_settings_label)
    c.show(show_ports=True)
