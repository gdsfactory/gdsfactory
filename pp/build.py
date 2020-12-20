import itertools
import multiprocessing
import os
import re
import shutil
import sys
import time
from glob import glob
from multiprocessing import Pool
from subprocess import PIPE, Popen, check_call

from pp.components import component_factory
from pp.config import CONFIG, logging
from pp.doe import load_does


def run_python(filename):
    """ Run a python script and keep track of some context """
    logging.debug("Running `{}`.".format(filename))
    command = ["python", filename]

    # Run the process
    t = time.time()
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    stdout, _ = process.communicate()
    total_time = time.time() - t
    if process.returncode == 0:
        logging.info("v {} ({:.1f}s)".format(os.path.relpath(filename), total_time))
    else:
        logging.info(
            "! Error in {} {:.1f}s)".format(os.path.relpath(filename), total_time)
        )
        # message = "! Error in `{}`".format(basename(filename))
        # logging.error(message, exc_info=(Exception, stderr.strip(), None))
    if len(stdout.decode().strip()) > 0:
        logging.debug("Output of python {}:\n{}".format(filename, stdout.strip()))
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
    logging.info(
        "Building splits on {} threads. {} files to run.".format(
            multiprocessing.cpu_count(), len(all_files)
        )
    )
    logging.info(
        "Debug information at {}".format(
            os.path.relpath(os.path.join(CONFIG["log_directory"], "debug.log"))
        )
    )

    # Now run all the files in batches of $CPU_SIZE.
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        for filename, rc in pool.imap_unordered(run_python, all_files):
            logging.debug("Finished {} {}".format(filename, rc))

    # Report on what we did.
    devices = glob(os.path.join(CONFIG["gds_directory"], "*.gds"))
    countmsg = "There are now {} GDS files in {}.".format(
        len(devices), os.path.relpath(CONFIG["gds_directory"])
    )
    logging.info("Finished building devices. {}".format(countmsg))


def build_clean():
    """ Cleans generated files such as build/. """
    target = CONFIG["build_directory"]
    if os.path.exists(target):
        shutil.rmtree(target)
        print(("Deleted {}".format(os.path.abspath(target))))


def build_cache_pull():
    """ Pull devices from the cache """
    if CONFIG.get("cache_url"):
        logging.info("Loading devices from cache...")
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
        logging.info("Nothing to push")
        return

    if CONFIG.get("cache_url"):
        logging.info("Uploading devices to cache...")
        check_call(
            [
                "rsync",
                "-rv",
                CONFIG["build_directory"] + "/",
                CONFIG["cache_url"],
                "--delete",
            ]
        )


def _build_doe(doe_name, config, component_factory=component_factory):
    from pp.write_doe import write_doe

    doe = config["does"][doe_name]
    component_type = doe.get("component")
    component_function = component_factory[component_type]
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


def build_does(filepath, component_factory=component_factory):
    """ this function is depreacted

    Writes DOE settings from config.yml file and writes GDS into build_directory

    If you want to use cache use pp.generate_does instead

    Write For each DOE:

    - GDS
    - json metadata
    - ports CSV
    - markdown report, with DOE settings
    """

    does = load_does(filepath)
    doe_names = does.keys()

    doe_params = zip(
        doe_names, itertools.repeat(filepath), itertools.repeat(component_factory)
    )
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    p.starmap(_build_doe, doe_params)

    # for doe_name in doe_names:
    #     p = multiprocessing.Process(target=_build_doe, args=(doe_name, config, component_factory=component_factory))
    #     p.start()


if __name__ == "__main__":
    does_path = CONFIG["samples_path"] / "mask" / "does.yml"
    build_does(does_path)

    # run_python("name.py")
