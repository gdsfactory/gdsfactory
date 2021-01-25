""" Command line interface for gdsfactory
"""
import os
import pathlib
import re
import shlex
import subprocess
import time

import click
from click.core import Context, Option

import pp.build as pb
from pp import CONFIG, klive
from pp.config import logging, print_config
from pp.gdsdiff.gdsdiff import gdsdiff
from pp.install import install_gdsdiff, install_generic_tech, install_klive
from pp.layers import LAYER
from pp.mask.merge_json import merge_json
from pp.mask.merge_markdown import merge_markdown
from pp.mask.merge_test_metadata import merge_test_metadata
from pp.mask.write_labels import write_labels

# from pp.write_doe_from_yaml import write_doe_from_yaml
from pp.write_doe_from_yaml import import_custom_doe_factories

VERSION = "2.2.9"
log_directory = CONFIG.get("log_directory")
cwd = pathlib.Path.cwd()
LAYER_LABEL = LAYER.LABEL


def shorten_command(cmd):
    """ Shortens a command if possible"""
    match = re.search(r"(\w+\.(py|drc))", cmd)
    return match.group() if match else cmd


def run_command(command):
    """ Run a command and keep track of some context """
    logging.info("Running `{}`".format(command))

    # Run the process and handle errors
    time0 = time.time()
    process = subprocess.Popen(
        shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    total_time = time.time() - time0

    # Either show that ther was an error, or just leave it
    if process.returncode == 0:
        message = "`{}` ran without errors in {:.2f}s.".format(
            shorten_command(command), total_time
        )
        logging.info(message)
        if stdout.strip():
            message = "Output of `{}`:".format(shorten_command(command))
            logging.info(message)
            logging.info(stdout.strip(), extra={"raw": True})
    else:
        message = "Error in `{}`".format(shorten_command(command))
        logging.error(message)
        raw = stdout.strip() + "\n" + stderr.strip()
        logging.error(raw, extra={"raw": True})

    return command, process.returncode


def print_version(ctx: Context, param: Option, value: bool) -> None:
    """ Prints the version """
    if not value or ctx.resilient_parsing:
        return
    click.echo(VERSION)
    ctx.exit()


@click.command(name="delete")
@click.argument("logfile", default="main", required=False)
def log_delete(logfile):
    """ Deletes logs """
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
def config_get(key):
    """ Shows key values from CONFIG """
    print_config(key)


"""
MASKS
"""


@click.group()
def mask():
    """ Commands for building masks """
    pass


@click.command(name="clean")
@click.option("--force", "-f", default=False, help="Force deletion", is_flag=True)
def build_clean(force):
    """ Deletes the build folder and contents """
    message = "Delete {}. Are you sure?".format(CONFIG["build_directory"])
    if force or click.confirm(message, default=True):
        pb.build_clean()


@click.command(name="build_devices")
@click.argument("regex", required=False, default=".*")
def build_devices(regex):
    """ Build all devices described in devices/"""
    pb.build_devices(regex)


@click.command(name="build_does")
def build_does():
    """ Build does defined in doe.yml"""
    print("this is deprecated")
    import_custom_doe_factories()
    # write_doe_from_yaml()
    pb.build_does()


@click.command(name="write_metadata")
@click.argument("label_layer", required=False, default=LAYER_LABEL)
def mask_merge(label_layer):
    """ merge JSON/Markdown from build/devices into build/mask"""

    gdspath = CONFIG["mask_gds"]
    write_labels(gdspath=gdspath, label_layer=label_layer)

    merge_json()
    merge_markdown()
    merge_test_metadata(config_path=CONFIG["config_path"])


@click.command(name="write_labels")
@click.argument("gdspath", default=None)
@click.argument("label_layer", required=False, default=LAYER_LABEL)
def write_mask_labels(gdspath, label_layer):
    """Find test and measurement labels."""
    if gdspath is None:
        gdspath = CONFIG["mask_gds"]

    write_labels(gdspath=gdspath, label_layer=label_layer)


"""
EXTRA
"""


@click.command()
@click.argument("filename")
def show(filename):
    """Show a GDS file using klive """
    klive.show(filename)


@click.command()
@click.argument("gdspath1")
@click.argument("gdspath2")
def diff(gdspath1, gdspath2):
    """Show boolean difference between two GDS files."""
    import pp

    diff = gdsdiff(str(gdspath1), str(gdspath2))
    pp.show(diff)


@click.command()
def install():
    """Install Klive, gdsdiff and generic tech """
    install_generic_tech()
    install_klive()
    install_gdsdiff()


@click.command()
def test():
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
cli.add_command(show)
cli.add_command(test)
cli.add_command(install)
cli.add_command(diff)


if __name__ == "__main__":
    cli()
