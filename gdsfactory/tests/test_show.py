import pathlib

import gdsfactory as gf


def test_show():
    c = gf.components.straight()
    c.show(show_ports=True)


def test_show_with_explicit_dir(tmpdir):
    c = gf.components.straight()
    gf.show(c, gdsdir=tmpdir)
    expected_output_filename: pathlib.Path = tmpdir / f"{c.name}.gds"
    assert expected_output_filename.isfile()
