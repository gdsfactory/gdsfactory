"""Tests for components."""

import subprocess

import pytest

from gdsfactory.config import PATH

scripts_dir = PATH.module / "components"
COMPONENT_FILES = list(scripts_dir.glob("**/*.py"))

# parameterize the tests by stem to get better test names
COMPONENT_FILES_BY_STEM = {p.stem: p for p in COMPONENT_FILES}
SCRIPTS_TO_SKIP = {"text_freetype"}


@pytest.mark.components_run
@pytest.mark.parametrize("component_file", COMPONENT_FILES_BY_STEM.keys())
def test_components_files_execution(component_file: str) -> None:
    """Tests that all components files run without error.

    Args:
        component_file: the name (stem) of the component file.
    """
    if component_file in SCRIPTS_TO_SKIP:
        pytest.skip("This script is currently marked to be skipped.")

    script_path = COMPONENT_FILES_BY_STEM[component_file]
    try:
        subprocess.run(
            ["python", script_path], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script failed with error: {e.stderr}")


if __name__ == "__main__":
    pytest.main([__file__, "-n", "logical"])
