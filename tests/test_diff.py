from pathlib import Path
from typing import Any

import pytest

from gdsfactory.difftest import diff


def assert_xor_fails(
    ref_gds: Path,
    run_gds: Path,
    test_name: str,
    capsys: pytest.CaptureFixture[Any],
    layers_with_xor: list[str],
) -> None:
    assert diff(ref_gds, run_gds, xor=True, test_name=test_name) is True
    captured = capsys.readouterr()
    for layer in layers_with_xor:
        assert "XOR difference on layer" in captured.out, captured.out
        assert f"{layer}" in captured.out, captured.out


_gds_dir = Path(__file__).parent / "gds"


def test_xor1(capsys: pytest.CaptureFixture[Any]) -> None:
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


def test_xor2(capsys: pytest.CaptureFixture[Any]) -> None:
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
