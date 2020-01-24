from click.testing import CliRunner
from pp.pf import cli
from pp import __version__


def test_pf_version():
    """ checks that the CLI returns the correct version """
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "version"])

    print(result.output)
    print(__version__)
    assert result.exit_code == 0
    assert result.output.startswith(__version__)


if __name__ == "__main__":
    test_pf_version()
