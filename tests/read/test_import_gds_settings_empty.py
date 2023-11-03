import gdsfactory as gf


def test_import_gds_settings_empty() -> None:
    gf.import_gds(gf.config.PATH.gdsdir / "mmi1x2.gds", read_metadata=True)


if __name__ == "__main__":
    test_import_gds_settings_empty()
