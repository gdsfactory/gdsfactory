import gdsfactory as gf


def test_clean_path_wrong() -> None:
    c = gf.Component("hi:there")
    gdspath = c.write_gds()
    gdspath = str(gdspath)
    assert ":" not in gdspath, gdspath


def test_clean_path_correct() -> None:
    c = gf.Component()
    gdspath = c.write_gds(gdspath="/tmp/hithere.gds")
    gdspath = str(gdspath)
    assert gdspath == "/tmp/hithere.gds", gdspath

    gdspath = c.write_gds(gdspath="/tmp/hithere*.gds")
    gdspath = str(gdspath)
    assert gdspath == "/tmp/hithere_.gds", gdspath


if __name__ == "__main__":
    test_clean_path_correct()
