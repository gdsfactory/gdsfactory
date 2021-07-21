""" Command line interface for gdsfactory
"""
import os
import pathlib
import shlex
import subprocess
from typing import Optional

import click
from click.core import Context, Option

import pp
import pp.build as pb
from pp.config import CONFIG, print_config
from pp.gdsdiff.gdsdiff import gdsdiff
from pp.import_gds import write_cells as write_cells_to_separate_gds
from pp.install import install_gdsdiff, install_generic_tech, install_klive
from pp.layers import LAYER, lyp_to_dataclass
from pp.mask.merge_json import merge_json
from pp.mask.merge_markdown import merge_markdown
from pp.mask.merge_test_metadata import merge_test_metadata
from pp.mask.write_labels import write_labels
from pp.types import PathType

# from pp.write_doe_from_yaml import write_doe_from_yaml
from pp.write_doe_from_yaml import import_custom_doe_factories

VERSION = "2.6.8"
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
GDS
"""


@click.group()
def gds() -> None:
    """Commands for dealing with GDS files"""
    pass


@click.command(name="layermap_to_dataclass")
@click.argument("filepath")
@click.option("--force", "-f", default=False, help="Force deletion", is_flag=True)
def layermap_to_dataclass(filepath, force: bool) -> None:
    """Converts klayout LYP to a dataclass"""
    filepath_lyp = pathlib.Path(filepath)
    filepath_py = filepath_lyp.with_suffix(".py")
    if not filepath_lyp.exists():
        raise FileNotFoundError(f"{filepath_lyp} not found")
    if not force and filepath_py.exists():
        raise FileExistsError(f"found {filepath_py}")
    lyp_to_dataclass(lyp_filepath=filepath_lyp)


@click.command(name="write_cells")
@click.argument("gdspath")
def write_cells(gdspath) -> None:
    """Write each all level cells into separate GDS files."""
    write_cells_to_separate_gds(gdspath)


@click.command(name="merge_gds_from_directory")
@click.argument("dirpath", required=False, default=None)
@click.argument("gdspath", required=False, default=None)
def merge_gds_from_directory(
    dirpath: Optional[PathType] = None, gdspath: Optional[PathType] = None
) -> None:
    """Merges GDS cells from a directory into a single GDS."""
    dirpath = dirpath or pathlib.Path.cwd()
    gdspath = gdspath or pathlib.Path.cwd() / "merged.gds"
    c = pp.component_from.gdsdir(dirpath=dirpath)
    c.write_gds(gdspath=gdspath)
    c.show()


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
@click.argument("yamlpath")
def build_does(yamlpath: str) -> None:
    """Build does defined in doe.yml"""
    print("this is deprecated")
    import_custom_doe_factories()
    # write_doe_from_yaml()
    pb.build_does(yamlpath)


@click.command(name="write_metadata")
@click.argument("label_layer", required=False, default=LAYER_LABEL)
def mask_merge(label_layer) -> None:
    """merge JSON/Markdown from build/devices into build/mask"""

    gdspath = CONFIG["mask_gds"]
    write_labels(gdspath=gdspath, label_layer=label_layer)

    merge_json()
    merge_markdown()
    merge_test_metadata(gdspath=gdspath)


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
@click.argument("filename")
def show(filename: str) -> None:
    """Show a GDS file using klive"""
    pp.show(filename)


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
    """`pf` is the photonics library command line tool.
    It helps to build, test, and configure masks and components.
    """
    pass


mask.add_command(build_clean)
mask.add_command(build_devices)
mask.add_command(build_does)
mask.add_command(mask_merge)
mask.add_command(write_mask_labels)

gds.add_command(layermap_to_dataclass)
gds.add_command(write_cells)
gds.add_command(merge_gds_from_directory)

cli.add_command(config_get)
cli.add_command(mask)
cli.add_command(show)
cli.add_command(test)
cli.add_command(install)
cli.add_command(diff)
cli.add_command(gds)


if __name__ == "__main__":
    cli()
