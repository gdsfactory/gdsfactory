from pathlib import Path

import pytest

from gdsfactory.difftest import diff


def assert_xor_fails(ref_gds, run_gds, test_name, capsys, layers_with_xor) -> None:
    assert diff(ref_gds, run_gds, xor=True, test_name=test_name) is True
    captured = capsys.readouterr()
    for layer in layers_with_xor:
        assert f"XOR difference on layer {layer}" in captured.out, captured.out


_gds_dir = Path(__file__).parent / "gds"


@pytest.mark.skip("TODO: fix")
def test_xor1(capsys) -> None:
    """Assert that the XOR flags the layer with A not B differences."""
    ref_gds = _gds_dir / "big_rect.gds"
    run_gds = _gds_dir / "small_rect.gds"
    assert_xor_fails(
        ref_gds=ref_gds,
        run_gds=run_gds,
        test_name="test_xor1",
        capsys=capsys,
        layers_with_xor=["2/0"],
    )


@pytest.mark.skip("TODO: fix")
def test_xor2(capsys) -> None:
    """Assert that the XOR flags the layer with B not A differences."""
    ref_gds = _gds_dir / "small_rect.gds"
    run_gds = _gds_dir / "big_rect.gds"
    assert_xor_fails(
        ref_gds=ref_gds,
        run_gds=run_gds,
        test_name="test_xor2",
        capsys=capsys,
        layers_with_xor=["2/0"],
    )
