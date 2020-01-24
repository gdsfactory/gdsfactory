from glob import glob
import itertools
from subprocess import Popen, PIPE, check_call
import os
import sys
from multiprocessing import Pool
import multiprocessing
import shutil

import time
import re

from pp.components import component_type2factory
from pp.config import CONFIG, load_config
from pp.logger import LOGGER
from pp.doe import load_does


def run_python(filename):
    """ Run a python script and keep track of some context """
    LOGGER.debug("Running `{}`.".format(filename))
    command = ["python", filename]

    # Run the process
    t = time.time()
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    stdout, _ = process.communicate()
    total_time = time.time() - t
    if process.returncode == 0:
        LOGGER.info("v {} ({:.1f}s)".format(os.path.relpath(filename), total_time))
    else:
        LOGGER.info(
            "! Error in {} {:.1f}s)".format(os.path.relpath(filename), total_time)
        )
        # message = "! Error in `{}`".format(basename(filename))
        # LOGGER.error(message, exc_info=(Exception, stderr.strip(), None))
    if len(stdout.decode().strip()) > 0:
        LOGGER.debug("Output of python {}:\n{}".format(filename, stdout.strip()))
    return filename, process.returncode


def build_devices(regex=".*", overwrite=True):
    """ Builds all the python files in devices/ """
    # Avoid accidentally rebuilding devices
    if (
        os.path.isdir(CONFIG["gds_directory"])
        and os.listdir(CONFIG["gds_directory"])
        and not overwrite
    ):
        print("Run `make clean` to remove already built devices.")
        sys.exit(0)

    # Collect all the files to run.
    all_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(CONFIG["devices_directory"])
        for f in filenames
        if os.path.splitext(f)[1] == ".py"
    ]
    all_files = sorted(all_files)
    all_files = [f for f in all_files if re.search(regex, f)]

    # Notify user
    LOGGER.info(
        "Building splits on {} threads. {} files to run.".format(
            multiprocessing.cpu_count(), len(all_files)
        )
    )
    LOGGER.info(
        "Debug information at {}".format(
            os.path.relpath(os.path.join(CONFIG["log_directory"], "debug.log"))
        )
    )

    # Now run all the files in batches of $CPU_SIZE.
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        for filename, rc in pool.imap_unordered(run_python, all_files):
            LOGGER.debug("Finished {} {}".format(filename, rc))

    # Report on what we did.
    devices = glob(os.path.join(CONFIG["gds_directory"], "*.gds"))
    countmsg = "There are now {} GDS files in {}.".format(
        len(devices), os.path.relpath(CONFIG["gds_directory"])
    )
    LOGGER.info("Finished building devices. {}".format(countmsg))


def build_clean():
    """ Cleans generated files such as build/. """
    target = CONFIG["build_directory"]
    if os.path.exists(target):
        shutil.rmtree(target)
        print(("Deleted {}".format(os.path.abspath(target))))


def build_cache_pull():
    """ Pull devices from the cache """
    if CONFIG.get("cache_url"):
        LOGGER.info("Loading devices from cache...")
        check_call(
            [
                "rsync",
                "-rv",
                "--delete",
                CONFIG["cache_url"],
                CONFIG["build_directory"] + "/",
            ]
        )


def build_cache_push():
    """ Push devices to the cache """
    if not os.listdir(CONFIG["build_directory"]):
        LOGGER.info("Nothing to push")
        return

    if CONFIG.get("cache_url"):
        LOGGER.info("Uploading devices to cache...")
        check_call(
            [
                "rsync",
                "-rv",
                CONFIG["build_directory"] + "/",
                CONFIG["cache_url"],
                "--delete",
            ]
        )


def _build_doe(doe_name, config, component_type2factory=component_type2factory):
    from pp.write_doe import write_doe

    doe = config["does"][doe_name]
    component_type = doe.get("component")
    component_function = component_type2factory[component_type]
    write_doe(
        component_type=component_function,
        doe_name=doe_name,
        do_permutations=doe.get("do_permutations", True),
        list_settings=doe.get("settings"),
        description=doe.get("description"),
        analysis=doe.get("analysis"),
        test=doe.get("test"),
        functions=doe.get("functions"),
    )


def build_does(config=CONFIG, component_type2factory=component_type2factory):
    """ Writes DOE settings from config.yml file and writes GDS into build_directory

    If you want to use cache use pp.generate_does instead

    Write For each DOE:

    - GDS
    - json metadata
    - ports CSV
    - markdown report, with DOE settings
    """
    if config.get("does") is None:
        raise ValueError(f"no does defined in {CONFIG}")

    does = load_does(config)
    doe_names = does.keys()

    doe_params = zip(
        doe_names, itertools.repeat(config), itertools.repeat(component_type2factory)
    )
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    p.starmap(_build_doe, doe_params)

    # for doe_name in doe_names:
    #     p = multiprocessing.Process(target=_build_doe, args=(doe_name, config, component_type2factory=component_type2factory))
    #     p.start()


if __name__ == "__main__":
    CONFIG = load_config(CONFIG["samples_path"] / "mask" / "config.yml")
    build_does(CONFIG)

    # run_python("name.py")
