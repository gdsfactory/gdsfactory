"""Command line interface. Type `gf` into a terminal."""

import os
import pathlib
import shlex
import subprocess
from typing import Optional

import click
from click.core import Context, Option

import gdsfactory
from gdsfactory.config import CONFIG, cwd, print_config
from gdsfactory.gdsdiff.gdsdiff import gdsdiff
from gdsfactory.install import install_gdsdiff, install_klayout_package
from gdsfactory.layers import lyp_to_dataclass
from gdsfactory.tech import LAYER
from gdsfactory.types import PathType
from gdsfactory.write_cells import write_cells as write_cells_to_separate_gds

VERSION = "5.41.0"
log_directory = CONFIG.get("log_directory")
LAYER_LABEL = LAYER.LABEL


def print_version(ctx: Context, param: Option, value: bool) -> None:
    """Prints the version."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(VERSION)
    ctx.exit()


@click.command(name="delete")
@click.argument("logfile", default="main", required=False)
def log_delete(logfile: str) -> None:
    """Deletes logs."""
    if not os.path.exists(log_directory):
        print("No logs found.")
        return

    filename = os.path.join(log_directory, f"{logfile}.log")
    subprocess.check_output(["rm", filename])


# TOOL


@click.group()
def tool() -> None:
    """Commands working with gdsfactory tool."""
    pass


@click.command(name="config")
@click.argument("key", required=False, default=None)
def config_get(key: str) -> None:
    """Shows key values from CONFIG."""
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
    """Converts klayout LYP to a dataclass."""
    filepath_lyp = pathlib.Path(filepath)
    filepath_py = filepath_lyp.with_suffix(".py")
    if not filepath_lyp.exists():
        raise FileNotFoundError(f"{filepath_lyp} not found")
    if not force and filepath_py.exists():
        raise FileExistsError(f"found {filepath_py}")
    lyp_to_dataclass(lyp_filepath=filepath_lyp)


@click.command(name="write_cells")
@click.argument("gdspath")
@click.argument("dirpath", required=False, default=None)
def write_cells(gdspath, dirpath) -> None:
    """Write each all level cells into separate GDS files."""
    write_cells_to_separate_gds(gdspath=gdspath, dirpath=dirpath)


@click.command(name="merge_gds")
@click.argument("dirpath", required=False, default=None)
@click.argument("gdspath", required=False, default=None)
def merge_gds(
    dirpath: Optional[PathType] = None, gdspath: Optional[PathType] = None
) -> None:
    """Merges GDS cells from a directory into a single GDS."""
    dirpath = dirpath or pathlib.Path.cwd()
    gdspath = gdspath or pathlib.Path.cwd() / "merged.gds"

    dirpath = pathlib.Path(dirpath)

    c = gdsfactory.read.from_gdsdir(dirpath=dirpath)
    c.write_gds(gdspath=gdspath)
    c.show(show_ports=True)


# @click.group()
# def watch() -> None:
#     """Watch YAML or python files."""
#     pass
# @click.option("--debug", "-d", default=False, help="debug", is_flag=True)
# @click.command()
# def webapp(debug: bool = False) -> None:
#     """Opens YAML based webapp."""
#     from gdsfactory.icyaml import app

#     if debug:
#         app.run_debug()

#     else:
#         app.run()


@click.argument("path", type=click.Path(exists=True), required=False, default=cwd)
@click.command()
def watch(path=cwd) -> None:
    """Filewatch YAML file."""
    from gdsfactory.watch import watch

    path = pathlib.Path(path)
    path = path.parent if path.is_dir() else path
    watch(str(path))


# EXTRA
@click.command()
@click.argument("filename")
def show(filename: str) -> None:
    """Show a GDS file using klive."""
    gdsfactory.show(filename)


@click.command()
@click.argument("gdspath1")
@click.argument("gdspath2")
@click.option("--xor", "-x", default=False, help="include xor", is_flag=True)
def diff(gdspath1: str, gdspath2: str, xor: bool = False) -> None:
    """Show boolean difference between two GDS files."""
    diff = gdsdiff(gdspath1, gdspath2, xor=xor)
    diff.show()


@click.command()
def install() -> None:
    """Install Klive and generic tech layermap."""
    install_klayout_package()
    install_gdsdiff()


@click.command(name="test")
def run_tests() -> None:
    """Run tests using pytest.

    You can also just run `pytest` directly.

    """
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
def gf() -> None:
    """`gf` is the gdsfactory command line tool."""


gds.add_command(layermap_to_dataclass)
gds.add_command(write_cells)
gds.add_command(merge_gds)
gds.add_command(show)
gds.add_command(diff)

tool.add_command(config_get)
tool.add_command(run_tests)
tool.add_command(install)

# yaml.add_command(webapp)
# watch.add_command(watch_yaml)

gf.add_command(gds)
gf.add_command(tool)
gf.add_command(watch)


if __name__ == "__main__":
    gf()
