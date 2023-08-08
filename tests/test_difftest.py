from pathlib import Path

import gdsfactory as gf


def test_difftest():
    c = gf.c.straight()
    test_dir = Path(__file__).parent / "test_difftest"
    run_dir = test_dir / "run"
    ref_dir = test_dir / "ref"

    c.write_gds(gdsdir=ref_dir)

    # assert that difftest runs without error for files which should be the same
    gf.difftest(c, dirpath=ref_dir, dirpath_run=run_dir)
    expected_run_filename = run_dir / f"{c.name}.gds"
    # assert that the run file was created where specified
    assert expected_run_filename.is_file()
