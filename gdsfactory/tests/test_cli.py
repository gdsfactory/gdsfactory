from click.testing import CliRunner

from gdsfactory import __version__
from gdsfactory.gf import gf


def test_cli() -> None:
    """checks that the CLI returns the correct version."""
    runner = CliRunner()
    result = runner.invoke(gf, ["--version"])

    # print(result.output)
    # print(__version__)
    assert result.exit_code == 0
    assert result.output.startswith(__version__)


if __name__ == "__main__":
    test_cli()
