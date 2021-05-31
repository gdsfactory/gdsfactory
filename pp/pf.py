""" Command line interface for gdsfactory
"""
import os
import pathlib
import shlex
import subprocess

import click
from click.core import Context, Option

import pp
import pp.build as pb
from pp import CONFIG, klive
from pp.config import print_config
from pp.gdsdiff.gdsdiff import gdsdiff
from pp.install import install_gdsdiff, install_generic_tech, install_klive
from pp.layers import LAYER
from pp.mask.merge_json import merge_json
from pp.mask.merge_markdown import merge_markdown
from pp.mask.merge_test_metadata import merge_test_metadata
from pp.mask.write_labels import write_labels

# from pp.write_doe_from_yaml import write_doe_from_yaml
from pp.write_doe_from_yaml import import_custom_doe_factories

VERSION = "2.5.2"
log_directory = CONFIG.get("log_directory")
cwd = pathlib.Path.cwd()
LAYER_LABEL = LAYER.LABEL


def print_version(ctx: Context, param: Option, value: bool) -> None:
    """Prints the version"""
    if not value or ctx.resilient_parsing:
        return
    click.echo(VERSION)
    ctx.exit()


@click.command(name="delete")
@click.argument("logfile", default="main", required=False)
def log_delete(logfile: str) -> None:
    """Deletes logs"""
    if not os.path.exists(log_directory):
        print("No logs found.")
        return

    filename = os.path.join(log_directory, "{}.log".format(logfile))
    subprocess.check_output(["rm", filename])


"""
CONFIG
"""


@click.command(name="config")
@click.argument("key", required=False, default=None)
def config_get(key: str) -> None:
    """Shows key values from CONFIG"""
    print_config(key)


"""
MASKS
"""


@click.group()
def mask() -> None:
    """Commands for building masks"""
    pass


@click.command(name="clean")
@click.option("--force", "-f", default=False, help="Force deletion", is_flag=True)
def build_clean(force: bool) -> None:
    """Deletes the build folder and contents"""
    message = "Delete {}. Are you sure?".format(CONFIG["build_directory"])
    if force or click.confirm(message, default=True):
        pb.build_clean()


@click.command(name="build_devices")
@click.argument("regex", required=False, default=".*")
def build_devices(regex) -> None:
    """Build all devices described in devices/"""
    pb.build_devices(regex)


@click.command(name="build_does")
def build_does() -> None:
    """Build does defined in doe.yml"""
    print("this is deprecated")
    import_custom_doe_factories()
    # write_doe_from_yaml()
    pb.build_does()


@click.command(name="write_metadata")
@click.argument("label_layer", required=False, default=LAYER_LABEL)
def mask_merge(label_layer) -> None:
    """merge JSON/Markdown from build/devices into build/mask"""

    gdspath = CONFIG["mask_gds"]
    write_labels(gdspath=gdspath, label_layer=label_layer)

    merge_json()
    merge_markdown()
    merge_test_metadata(config_path=CONFIG["config_path"])


@click.command(name="write_labels")
@click.argument("gdspath", default=None)
@click.argument("label_layer", required=False, default=LAYER_LABEL)
def write_mask_labels(gdspath: str, label_layer) -> None:
    """Find test and measurement labels."""
    if gdspath is None:
        gdspath = CONFIG["mask_gds"]

    write_labels(gdspath=gdspath, label_layer=label_layer)


"""
EXTRA
"""


@click.command()
@click.argument("filepath")
def merge_cells(filepath: str) -> None:
    """Merge GDS cells into a top level."""
    filepath = pathlib.Path(filepath)
    filepath_out = filepath / "merged.gds"

    c = pp.Component("merge-cells")

    cells = filepath.glob("*.gds")
    for cell in cells:
        if not isinstance(cell, pp.Component):
            cell = pp.import_gds(cell)
        c << cell
    c.show()
    c.write_gds(filepath_out)


@click.command()
@click.argument("filename")
def show(filename: str) -> None:
    """Show a GDS file using klive"""
    klive.show(filename)


@click.command()
@click.argument("gdspath1")
@click.argument("gdspath2")
def diff(gdspath1: str, gdspath2: str) -> None:
    """Show boolean difference between two GDS files."""
    import pp

    diff = gdsdiff(str(gdspath1), str(gdspath2))
    pp.show(diff)


@click.command()
def install() -> None:
    """Install Klive, gdsdiff and generic tech"""
    install_generic_tech()
    install_klive()
    install_gdsdiff()


@click.command()
def test() -> None:
    """Run tests using pytest.
    Strictly speaking you should just run `pytest` directly."""

    os.chdir(CONFIG["repo_path"])
    command = shlex.split("pytest")
    subprocess.call(command)


@click.group()
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Show the version number.",
)
def cli():
    """`pf` is the photonics factory command line tool.
    It helps to build, test, and configure masks and components.
    """
    pass


mask.add_command(build_clean)
mask.add_command(build_devices)
mask.add_command(build_does)
mask.add_command(mask_merge)
mask.add_command(write_mask_labels)

cli.add_command(config_get)
cli.add_command(mask)
cli.add_command(merge_cells)
cli.add_command(show)
cli.add_command(test)
cli.add_command(install)
cli.add_command(diff)


if __name__ == "__main__":
    cli()
