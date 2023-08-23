import gdsfactory as gf


def test_clean_path() -> None:
    c = gf.Component("hi:there")
    gdspath = c.write_gds()
    gdspath = str(gdspath)
    assert ":" not in gdspath, gdspath


if __name__ == "__main__":
    test_clean_path()
