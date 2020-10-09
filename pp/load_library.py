import hashlib
import pathlib
import git
import time
import os
import pp
from pp.config import CONFIG, conf


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def list_gds_files(gdslib=CONFIG["gdslib"], tech=conf.tech.name):
    gdslib_path = pathlib.Path(gdslib) / tech
    return [gds for gds in gdslib_path.glob("*.gds")]


def pull_library(repo_path=CONFIG["gdslib"]):
    if os.path.isdir(repo_path):
        print("git pull: {}".format(repo_path))
        g = git.cmd.Git(repo_path)
        g.pull()


def load_library(gdslib=CONFIG["gdslib"], tech=conf.tech.name):
    """
    Load a cell library from the tech
    """
    lib = {}
    gds_files = list_gds_files(gdslib, tech)

    for filepath in gds_files:
        filename = filepath.stem
        lib[filename] = pp.load_component(
            name=filename, dirpath=filepath.parent, with_info_labels=False
        )

    return lib


def print_lib_info(gdslib=CONFIG["gdslib"], dump_file=None):
    """
    cell    cell_path   md5_hash
    """
    gds_files = list_gds_files(gdslib)
    if not gds_files:
        return

    names = []
    filepaths = []
    geohashes = []
    build_dates = []
    for filepath in gds_files:
        root, name = os.path.split(filepath)
        name = name[:-4]
        geohash = pp.load_component(name=name, dirpath=root).hash_geometry()
        names += [name]
        filepaths += [str(filepath)]
        geohashes += [geohash]
        build_dates += [time.ctime(os.path.getmtime(filepath))]

    n_name = max([len(x) for x in names]) + 1
    n_path = max([len(x) for x in filepaths]) + 1
    n_geohash = max([len(x) for x in geohashes]) + 1
    n_build_date = max([len(x) for x in build_dates]) + 1

    lines = []
    line = (
        "name".ljust(n_name)
        + "path".ljust(n_path)
        + "hash geometry".ljust(n_geohash)
        + "build date".ljust(n_build_date)
    )  # +'md5 hash'.ljust(n_hash)
    lines += [line]
    lines += ["-" * len(line)]
    for name, path, geohash, build_date in zip(
        names, filepaths, geohashes, build_dates
    ):
        line = "{}{}{}{}".format(
            name.ljust(n_name),
            path.ljust(n_path),
            geohash.ljust(n_geohash),
            build_date.ljust(n_build_date),
        )
        lines += [line]

    assert not dump_file.endswith(".gds")
    if dump_file is None:
        print()
        for line in lines:
            print(line)
        print()

    elif type(dump_file) == str:
        with open(dump_file, "w") as fw:
            for line in lines:
                fw.write(line + "\n")

    elif hasattr(dump_file, "write"):
        for line in lines:
            dump_file.write(line + "\n")
    else:
        raise ValueError("invalid filepath, {}".format(dump_file))


if __name__ == "__main__":
    print(CONFIG["gdslib"])
    print(list_gds_files())
    lib = load_library()
    print(lib.keys())
