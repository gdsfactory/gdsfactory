import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from gdsfactory.cli import app


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_gds_file(temp_dir: Path) -> Path:
    """Create a sample GDS file for testing."""
    import gdsfactory as gf

    c = gf.components.rectangle()
    gds_path = temp_dir / "sample.gds"
    c.write_gds(gds_path)
    return gds_path


@pytest.fixture
def sample_lyp_file(temp_dir: Path) -> Path:
    """Create a sample LYP file for testing."""
    lyp_content = """<?xml version="1.0" encoding="utf-8"?>
<layer-properties>
  <properties>
    <frame-color>#ff0000</frame-color>
    <fill-color>#ff0000</fill-color>
    <frame-brightness>0</frame-brightness>
    <fill-brightness>0</fill-brightness>
    <dither-pattern>I9</dither-pattern>
    <line-style/>
    <valid>true</valid>
    <visible>true</visible>
    <transparent>false</transparent>
    <width>1</width>
    <marked>false</marked>
    <xfill>false</xfill>
    <animation>0</animation>
    <name>layer1/0@*</name>
    <source>1/0@*</source>
  </properties>
</layer-properties>"""
    lyp_path = temp_dir / "sample.lyp"
    lyp_path.write_text(lyp_content)
    return lyp_path


class TestCLIHelp:
    """Test CLI help functionality."""

    def test_app_help(self, runner: CliRunner) -> None:
        """Test main help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "Commands" in result.stdout

    def test_individual_command_help(self, runner: CliRunner) -> None:
        """Test help for individual commands."""
        commands = [
            "layermap-to-dataclass",
            "write-cells",
            "merge-gds",
            "watch",
            "show",
            "gds-diff",
            "install-klayout-genericpdk",
            "install-git-diff",
            "version",
            "from-updk",
        ]

        for command in commands:
            result = runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0, f"Help failed for command: {command}"
            assert "Usage:" in result.stdout


class TestLayermapToDataclass:
    """Test layermap-to-dataclass command."""

    def test_layermap_to_dataclass_success(
        self, runner: CliRunner, sample_lyp_file: Path, temp_dir: Path
    ) -> None:
        """Test successful conversion of LYP to dataclass."""
        with patch("gdsfactory.technology.lyp_to_dataclass") as mock_convert:
            result = runner.invoke(app, ["layermap-to-dataclass", str(sample_lyp_file)])
            assert result.exit_code == 0
            mock_convert.assert_called_once()

    def test_layermap_to_dataclass_file_not_found(self, runner: CliRunner) -> None:
        """Test error when LYP file doesn't exist."""
        result = runner.invoke(app, ["layermap-to-dataclass", "nonexistent.lyp"])
        assert result.exit_code == 1
        # Check that the exception was raised (typer catches it and sets exit code to 1)
        assert "FileNotFoundError" in str(result.exception) or "not found" in str(
            result.exception
        )

    def test_layermap_to_dataclass_file_exists_no_force(
        self, runner: CliRunner, sample_lyp_file: Path
    ) -> None:
        """Test error when output file exists and --force not used."""
        py_file = sample_lyp_file.with_suffix(".py")
        py_file.write_text("# existing file")

        result = runner.invoke(app, ["layermap-to-dataclass", str(sample_lyp_file)])
        assert result.exit_code == 1
        # Check that the exception was raised
        assert "FileExistsError" in str(result.exception) or "found" in str(
            result.exception
        )

    def test_layermap_to_dataclass_force_overwrite(
        self, runner: CliRunner, sample_lyp_file: Path
    ) -> None:
        """Test successful overwrite with --force flag."""
        py_file = sample_lyp_file.with_suffix(".py")
        py_file.write_text("# existing file")

        with patch("gdsfactory.technology.lyp_to_dataclass") as mock_convert:
            result = runner.invoke(
                app, ["layermap-to-dataclass", str(sample_lyp_file), "--force"]
            )
            assert result.exit_code == 0
            mock_convert.assert_called_once()


class TestWriteCells:
    """Test write-cells command."""

    def test_write_cells_recursive(
        self, runner: CliRunner, sample_gds_file: Path, temp_dir: Path
    ) -> None:
        """Test write-cells with recursive option."""
        output_dir = temp_dir / "output"

        with patch("gdsfactory.write_cells.write_cells_recursively") as mock_write:
            result = runner.invoke(
                app,
                [
                    "write-cells",
                    str(sample_gds_file),
                    "--dirpath",
                    str(output_dir),
                    "--recursively",
                ],
            )
            assert result.exit_code == 0
            mock_write.assert_called_once_with(
                gdspath=str(sample_gds_file), dirpath=str(output_dir)
            )

    def test_write_cells_non_recursive(
        self, runner: CliRunner, sample_gds_file: Path, temp_dir: Path
    ) -> None:
        """Test write-cells without recursive option."""
        output_dir = temp_dir / "output"

        with patch("gdsfactory.write_cells.write_cells") as mock_write:
            result = runner.invoke(
                app,
                [
                    "write-cells",
                    str(sample_gds_file),
                    "--dirpath",
                    str(output_dir),
                    "--no-recursively",
                ],
            )
            assert result.exit_code == 0
            mock_write.assert_called_once_with(
                gdspath=str(sample_gds_file), dirpath=str(output_dir)
            )


class TestMergeGds:
    """Test merge-gds command."""

    def test_merge_gds_default_paths(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test merge-gds with default paths."""
        # Create some sample GDS files in the directory
        import gdsfactory as gf

        for i in range(2):
            c = gf.components.rectangle(size=(i + 1, i + 1))
            c.write_gds(temp_dir / f"rect_{i}.gds")

        with patch("gdsfactory.read.from_gdspaths.from_gdsdir") as mock_from_gdsdir:
            mock_component = Mock()
            mock_from_gdsdir.return_value = mock_component

            # Change to temp directory to test default behavior
            original_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                result = runner.invoke(app, ["merge-gds"])
                assert result.exit_code == 0
                mock_from_gdsdir.assert_called_once()
                mock_component.write_gds.assert_called_once()
                mock_component.show.assert_called_once()
            finally:
                os.chdir(original_cwd)

    def test_merge_gds_custom_paths(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test merge-gds with custom input and output paths."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_file = temp_dir / "merged_custom.gds"

        with patch("gdsfactory.read.from_gdspaths.from_gdsdir") as mock_from_gdsdir:
            mock_component = Mock()
            mock_from_gdsdir.return_value = mock_component

            result = runner.invoke(
                app,
                [
                    "merge-gds",
                    "--dirpath",
                    str(input_dir),
                    "--gdspath",
                    str(output_file),
                ],
            )
            assert result.exit_code == 0
            mock_from_gdsdir.assert_called_once_with(dirpath=input_dir)
            mock_component.write_gds.assert_called_once_with(gdspath=output_file)


class TestWatch:
    """Test watch command."""

    def test_watch_default_options(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test watch command with default options."""
        with patch("gdsfactory.cli._watch") as mock_watch:
            result = runner.invoke(app, ["watch", str(temp_dir)])
            assert result.exit_code == 0
            mock_watch.assert_called_once_with(
                str(temp_dir.absolute()),
                pdk=None,
                run_main=False,
                run_cells=False,
                pre_run=False,
            )

    def test_watch_with_options(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test watch command with various options."""
        with patch("gdsfactory.cli._watch") as mock_watch:
            with patch("gdsfactory.CONF") as mock_conf:
                result = runner.invoke(
                    app,
                    [
                        "watch",
                        str(temp_dir),
                        "--pdk",
                        "test_pdk",
                        "--run-main",
                        "--run-cells",
                        "--pre-run",
                        "--overwrite",
                    ],
                )
                assert result.exit_code == 0
                mock_watch.assert_called_once_with(
                    str(temp_dir.absolute()),
                    pdk="test_pdk",
                    run_main=True,
                    run_cells=True,
                    pre_run=True,
                )
                assert mock_conf.cell_overwrite_existing is True


class TestShow:
    """Test show command."""

    def test_show_gds_file(self, runner: CliRunner, sample_gds_file: Path) -> None:
        """Test showing a GDS file."""
        with patch("gdsfactory.cli._show") as mock_show:
            result = runner.invoke(app, ["show", str(sample_gds_file)])
            assert result.exit_code == 0
            mock_show.assert_called_once_with(str(sample_gds_file))


class TestGdsDiff:
    """Test gds-diff command."""

    def test_gds_diff_basic(
        self, runner: CliRunner, sample_gds_file: Path, temp_dir: Path
    ) -> None:
        """Test basic GDS diff functionality."""
        # Create a second GDS file
        import gdsfactory as gf

        c2 = gf.components.rectangle(size=(2, 2))
        gds2_path = temp_dir / "sample2.gds"
        c2.write_gds(gds2_path)

        with patch("gdsfactory.cli.diff") as mock_diff:
            result = runner.invoke(
                app, ["gds-diff", str(sample_gds_file), str(gds2_path)]
            )
            assert result.exit_code == 0
            mock_diff.assert_called_once_with(
                str(sample_gds_file), str(gds2_path), xor=False
            )

    def test_gds_diff_with_xor(
        self, runner: CliRunner, sample_gds_file: Path, temp_dir: Path
    ) -> None:
        """Test GDS diff with XOR option."""
        import gdsfactory as gf

        c2 = gf.components.rectangle(size=(2, 2))
        gds2_path = temp_dir / "sample2.gds"
        c2.write_gds(gds2_path)

        with patch("gdsfactory.cli.diff") as mock_diff:
            result = runner.invoke(
                app, ["gds-diff", str(sample_gds_file), str(gds2_path), "--xor"]
            )
            assert result.exit_code == 0
            mock_diff.assert_called_once_with(
                str(sample_gds_file), str(gds2_path), xor=True
            )


class TestInstallCommands:
    """Test installation commands."""

    def test_install_klayout_genericpdk(self, runner: CliRunner) -> None:
        """Test installing KLayout generic PDK."""
        with patch("gdsfactory.cli.install_klayout_package") as mock_install:
            result = runner.invoke(app, ["install-klayout-genericpdk"])
            assert result.exit_code == 0
            mock_install.assert_called_once()

    def test_install_git_diff(self, runner: CliRunner) -> None:
        """Test installing git diff."""
        with patch("gdsfactory.cli.install_gdsdiff") as mock_install:
            result = runner.invoke(app, ["install-git-diff"])
            assert result.exit_code == 0
            mock_install.assert_called_once()


class TestVersion:
    """Test version command."""

    def test_version_command(self, runner: CliRunner) -> None:
        """Test version command output."""
        with patch("gdsfactory.cli.print_version_plugins") as mock_print:
            result = runner.invoke(app, ["version"])
            assert result.exit_code == 0
            mock_print.assert_called_once()


class TestFromUpdk:
    """Test from-updk command."""

    def test_from_updk_basic(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test basic from-updk functionality."""
        input_file = temp_dir / "test.yaml"
        input_file.write_text("blocks: {}")

        with patch("gdsfactory.cli.from_updk") as mock_from_updk:
            result = runner.invoke(app, ["from-updk", str(input_file)])
            assert result.exit_code == 0
            expected_output = input_file.with_suffix(".py")
            mock_from_updk.assert_called_once_with(
                str(input_file), filepath_out=expected_output
            )

    def test_from_updk_custom_output(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test from-updk with custom output path."""
        input_file = temp_dir / "test.yaml"
        output_file = temp_dir / "custom_output.py"
        input_file.write_text("blocks: {}")

        with patch("gdsfactory.cli.from_updk") as mock_from_updk:
            result = runner.invoke(
                app, ["from-updk", str(input_file), "--output", str(output_file)]
            )
            assert result.exit_code == 0
            mock_from_updk.assert_called_once_with(
                str(input_file), filepath_out=output_file
            )


class TestBuildCommand:
    """Test build command from kfactory."""

    def test_build_command_exists(self, runner: CliRunner) -> None:
        """Test that build command is available."""
        result = runner.invoke(app, ["build", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout


class TestCLIEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_command(self, runner: CliRunner) -> None:
        """Test behavior with invalid command."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0

    def test_missing_required_args(self, runner: CliRunner) -> None:
        """Test behavior when required arguments are missing."""
        # Test commands that require arguments
        commands_requiring_args = [
            "layermap-to-dataclass",
            "write-cells",
            "show",
            "gds-diff",
            "from-updk",
        ]

        for command in commands_requiring_args:
            result = runner.invoke(app, [command])
            assert result.exit_code != 0, (
                f"Command {command} should fail without required args"
            )


# Test CLI path handling fixes
class TestCLIPathHandling:
    """Test that path handling issues are fixed."""

    def test_merge_gds_empty_string_paths(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test that empty strings are handled correctly in merge_gds."""
        import gdsfactory as gf

        # Create a test GDS file in temp directory
        c = gf.components.rectangle()
        c.write_gds(temp_dir / "test.gds")

        with patch("gdsfactory.read.from_gdspaths.from_gdsdir") as mock_from_gdsdir:
            mock_component = Mock()
            mock_from_gdsdir.return_value = mock_component

            # Change to temp directory
            original_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                # Test with empty strings (should use current directory)
                result = runner.invoke(
                    app, ["merge-gds", "--dirpath", "", "--gdspath", ""]
                )
                assert result.exit_code == 0
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__])
