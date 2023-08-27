import gdsfactory as gf


def test_clean_path_wrong() -> None:
    c = gf.Component()
    c.name = "Strange:Characters:in:the:filename"
    gdspath = c.write_gds()
    gdspath = str(gdspath)
    assert ":" not in gdspath, gdspath


if __name__ == "__main__":
    test_clean_path_wrong()
