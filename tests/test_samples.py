"""Tests for sample scripts."""

import subprocess

import pytest

from gdsfactory.config import PATH

scripts_dir = PATH.module / "samples"
PYTHON_SAMPLE_SCRIPTS = list(scripts_dir.glob("*.py"))

# parameterize the tests by stem to get better test names
PYTHON_SAMPLE_SCRIPTS_BY_STEM = {p.stem: p for p in PYTHON_SAMPLE_SCRIPTS}
SCRIPTS_TO_SKIP: set[str] = set()


@pytest.mark.parametrize("script_name", PYTHON_SAMPLE_SCRIPTS_BY_STEM.keys())
def test_script_execution(script_name: str) -> None:
    """Tests that all python sample scripts run without error.

    Args:
        script_name: the name (stem) of the python script.
    """
    if script_name in SCRIPTS_TO_SKIP:
        pytest.skip("This script is currently marked to be skipped.")

    script_path = PYTHON_SAMPLE_SCRIPTS_BY_STEM[script_name]
    try:
        subprocess.run(
            ["python", script_path], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script failed with error: {e.stderr}")
