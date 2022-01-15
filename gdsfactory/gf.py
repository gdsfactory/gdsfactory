""" Command line interface for gdsfactory."""
import os
import pathlib
import shlex
import subprocess
from typing import Optional

import click
from click.core import Context, Option

import gdsfactory
import gdsfactory.build as pb
from gdsfactory.config import CONFIG, print_config
from gdsfactory.gdsdiff.gdsdiff import gdsdiff
from gdsfactory.install import install_gdsdiff, install_generic_tech, install_klive
from gdsfactory.layers import lyp_to_dataclass
from gdsfactory.mask.merge_json import merge_json
from gdsfactory.mask.merge_markdown import merge_markdown
from gdsfactory.mask.merge_test_metadata import merge_test_metadata
from gdsfactory.mask.write_labels import write_labels

# from gdsfactory.write_doe_from_yaml import write_doe_from_yaml
from gdsfactory.sweep.write_sweep_from_yaml import import_custom_doe_factories
from gdsfactory.tech import LAYER
from gdsfactory.types import PathType
from gdsfactory.write_cells import write_cells as write_cells_to_separate_gds

VERSION = "3.10.3"
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


# TOOL


@click.group()
def tool() -> None:
    """Commands working with gdsfactory tool"""
    pass


@click.command(name="config")
@click.argument("key", required=False, default=None)
def config_get(key: str) -> None:
    """Shows key values from CONFIG"""
    print_config(key)


# GDS


@click.group()
def gds() -> None:
    """Commands for dealing with GDS files."""
    pass


@click.command(name="layermap_to_dataclass")
@click.argument("filepath", type=click.Path(exists=True))
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
    c = gdsfactory.read.gdsdir(dirpath=dirpath)
    c.write_gds(gdspath=gdspath)
    c.show()


# MASKS


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
    """Build does defined in sweep.yml"""
    print("this is deprecated")
    import_custom_doe_factories()
    # write_doe_from_yaml()
    pb.build_does(yamlpath)


@click.command(name="write_metadata")
@click.argument("layer_label", required=False, default=LAYER_LABEL)
def mask_merge(layer_label) -> None:
    """merge JSON/Markdown from build/devices into build/mask"""

    gdspath = CONFIG["mask_gds"]
    write_labels(gdspath=gdspath, layer_label=layer_label)

    merge_json()
    merge_markdown()
    merge_test_metadata(gdspath=gdspath)


@click.command(name="write_labels")
@click.argument("gdspath", default=None)
@click.argument("layer_label", required=False, default=LAYER_LABEL)
def write_mask_labels(gdspath: str, layer_label) -> None:
    """Find test and measurement labels."""
    if gdspath is None:
        gdspath = CONFIG["mask_gds"]

    write_labels(gdspath=gdspath, layer_label=layer_label)


# EXTRA


@click.command()
@click.argument("filename")
def show(filename: str) -> None:
    """Show a GDS file using klive"""
    gdsfactory.show(filename)


@click.command()
@click.argument("gdspath1")
@click.argument("gdspath2")
@click.option("--xor", "-x", default=False, help="include xor", is_flag=True)
def diff(gdspath1: str, gdspath2: str, xor: bool = False) -> None:
    """Show boolean difference between two GDS files."""
    diff = gdsdiff(str(gdspath1), str(gdspath2), xor=xor)
    diff.show()


@click.command()
def install() -> None:
    """Install Klive and generic tech layermap."""
    install_generic_tech()
    install_klive()
    install_gdsdiff()


@click.command()
def test() -> None:
    """Run tests using pytest.
    You can also just run `pytest` directly."""

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
def gf():
    """`gf` is the command line tool for gdsfactory.
    It helps you work with GDS files.
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
gds.add_command(show)
gds.add_command(diff)

tool.add_command(config_get)
tool.add_command(test)
tool.add_command(install)

gf.add_command(gds)
gf.add_command(tool)
gf.add_command(mask)


if __name__ == "__main__":
    gf()
