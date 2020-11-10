""" Command line interface for gdsfactory
"""

import os
import re
import shlex
import subprocess
import time
import pathlib
import click
import git

from pp import CONFIG
from pp.layers import LAYER
from pp.config import logging
from pp import klive
from pp.config import print_config

# from pp.write_doe_from_yaml import write_doe_from_yaml
from pp.write_doe_from_yaml import import_custom_doe_factories

from pp.mask.merge_json import merge_json
from pp.mask.merge_markdown import merge_markdown
from pp.mask.merge_test_metadata import merge_test_metadata
from pp.mask.write_labels import write_labels

import pp.build as pb

from pp.tests.test_factory import lock_components_with_changes


VERSION = "2.1.2"
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


def print_version(ctx, param, value):
    """ Prints the version """
    if not value or ctx.resilient_parsing:
        return
    click.echo(VERSION)
    ctx.exit()


"""
LOGS
"""


@click.group()
def log():
    """ Work with logs """
    pass


@click.command(name="show")
@click.argument("logfile", default="main", required=False)
def log_show(logfile):
    """ Show logs """
    if not os.path.exists(log_directory):
        print("No logs found.")
        return

    filename = os.path.join(log_directory, "{}.log".format(logfile))
    with open(filename) as f:
        data = f.read()
        print(data.strip() if data else "Logs are empty")


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
LIBRARY
"""


@click.group()
def library():
    """ Commands for managing libraries """
    pass


@click.argument("dirname", required=False, default=None)
@click.command(name="pull")
def library_pull(dirname):
    """ Pull the library repo """
    if dirname:
        repo_path = cwd / dirname
    else:
        repo_path = CONFIG["gdslib"]

    if os.path.isdir(repo_path):
        print("git pull: {}".format(repo_path))
        g = git.cmd.Git(repo_path)
        g.pull()


@click.command(name="lock")
@click.argument("dirname", required=False, default=None)
def library_lock(dirname):
    """ lock component with changes """

    if dirname:
        path_library = cwd / dirname
    else:
        path_library = CONFIG["gdslib"]

    if os.path.isdir(path_library):
        lock_components_with_changes(path_library=path_library)


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
    """ find test and measurement labels """
    if gdspath is None:
        gdspath = CONFIG["mask_gds"]

    write_labels(gdspath=gdspath, label_layer=label_layer)


"""
EXTRA
"""


@click.command()
@click.argument("filename")
def show(filename):
    """ Show a GDS file using KLive """
    klive.show(filename)


@click.command()
def test():
    """ Run tests using pytest.
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
    """ `pf` is the photonics factory command line tool.
    It helps to build, test, and configure masks and components.
    """
    pass


log.add_command(log_show)
log.add_command(log_delete)

library.add_command(library_pull)
library.add_command(library_lock)

mask.add_command(build_clean)
mask.add_command(build_devices)
mask.add_command(build_does)
mask.add_command(mask_merge)
mask.add_command(write_mask_labels)

cli.add_command(config_get)
cli.add_command(library)
cli.add_command(log)
cli.add_command(mask)
cli.add_command(show)
cli.add_command(test)


if __name__ == "__main__":
    cli()
