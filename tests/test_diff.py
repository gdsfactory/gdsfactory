from pathlib import Path
from typing import Any

import pytest

from gdsfactory.difftest import DiffResult, diff


def assert_xor_fails(
    ref_gds: Path,
    run_gds: Path,
    test_name: str,
    capsys: pytest.CaptureFixture[Any],
    layers_with_xor: list[str],
) -> None:
    assert diff(ref_gds, run_gds, xor=True, test_name=test_name)
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


def test_diff_returns_diff_result() -> None:
    """diff() returns a DiffResult, truthy when files differ, falsy when identical."""
    ref_gds = _gds_dir / "big_rect.gds"
    run_gds = _gds_dir / "small_rect.gds"

    result = diff(ref_gds, run_gds, test_name="test_result")
    assert isinstance(result, DiffResult)
    assert result.has_differences is True
    assert bool(result) is True

    same = diff(ref_gds, ref_gds, test_name="test_same")
    assert isinstance(same, DiffResult)
    assert same.has_differences is False
    assert bool(same) is False


def test_diff_layer_metrics() -> None:
    """Per-layer metrics are correct for big_rect vs small_rect (small fully inside big)."""
    ref_gds = _gds_dir / "big_rect.gds"
    run_gds = _gds_dir / "small_rect.gds"

    result = diff(ref_gds, run_gds, test_name="test_metrics")
    assert len(result.layers) == 1

    layer = result.layers[0]
    assert layer.layer == (2, 0)
    assert layer.present_in == "both"

    # big_rect: 1.55 x 1.28 = 1.984 µm², small_rect: 0.81 x 0.64 = 0.5184 µm²
    assert layer.ref_area == pytest.approx(1.984, abs=0.01)
    assert layer.run_area == pytest.approx(0.5184, abs=0.01)

    # small is fully inside big, so XOR = big - small = 1.4656 µm²
    assert layer.xor_area == pytest.approx(1.4656, abs=0.01)

    # IoU = intersection / union = small / big = 0.5184 / 1.984 ≈ 0.2613
    assert layer.iou == pytest.approx(0.2613, abs=0.01)

    # bbox of XOR region should cover the big rect extent
    assert layer.bbox is not None
    left, bottom, right, top = layer.bbox
    assert left <= right
    assert bottom <= top
    assert (right - left) == pytest.approx(1.55, abs=0.01)
    assert (top - bottom) == pytest.approx(1.28, abs=0.01)

    assert layer.polygon_count_ref == 1
    assert layer.polygon_count_run == 1


def test_diff_identical_files() -> None:
    """Identical files produce no-difference result with empty layers."""
    ref_gds = _gds_dir / "big_rect.gds"
    result = diff(ref_gds, ref_gds, test_name="test_identical")
    assert not result
    assert result.layers == []


def test_diff_missing_layer(tmp_path: Path) -> None:
    """A layer present in only one file gets IoU=0 and correct present_in."""
    import gdsfactory as gf

    # ref has layers (1,0) and (2,0); run has only (1,0)
    ref = gf.Component()
    ref.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=(1, 0))
    ref.add_polygon([(0, 0), (5, 0), (5, 5), (0, 5)], layer=(2, 0))
    ref_path = tmp_path / "ref.gds"
    ref.write_gds(ref_path)

    run = gf.Component()
    run.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=(1, 0))
    run_path = tmp_path / "run.gds"
    run.write_gds(run_path)

    result = diff(ref_path, run_path, test_name="test_missing")
    assert result

    missing = [ld for ld in result.layers if ld.layer == (2, 0)]
    assert len(missing) == 1
    layer = missing[0]
    assert layer.present_in == "ref_only"
    assert layer.iou == 0.0
    assert layer.run_area == 0.0
    assert layer.ref_area > 0
    assert layer.polygon_count_run == 0
    assert layer.polygon_count_ref == 1
