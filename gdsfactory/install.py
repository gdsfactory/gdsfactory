import configparser
import os
import pathlib
import shutil
import sys


def install_gdsdiff():
    home = pathlib.Path.home()
    git_config_path = home / ".gitconfig"
    git_attributes_path = home / ".gitattributes"

    if git_config_path.exists():
        git_config_str = open(git_config_path).read()
    else:
        git_config_str = "empty"

    if git_attributes_path.exists():
        git_attributes_str = open(git_attributes_path).read()
    else:
        git_attributes_str = "empty"

    if "gds_diff" not in git_config_str:
        print("gdsdiff shows boolean differences in Klayout")
        print("git diff FILE.GDS")
        print("Appending the gdsdiff command to your ~/.gitconfig")

        config = configparser.RawConfigParser()
        config.read(git_config_path)
        key = 'diff "gds_diff"'

        if key not in config.sections():
            config.add_section(key)
            config.set(key, "command", "python -m gdsdiff.gds_diff_git")
            config.set(key, "binary", "True")

            with open(git_config_path, "w+") as f:
                config.write(f, space_around_delimiters=True)

    if "gds_diff" not in git_attributes_str:
        print("Appending the gdsdiff command to your ~/.gitattributes")

        with open(git_attributes_path, "a") as f:
            f.write("*.gds diff=gds_diff\n")


def install_klive():
    if sys.platform == "win32":
        klayout_folder = "KLayout"
    else:
        klayout_folder = ".klayout"
    home = pathlib.Path.home()
    dest_folder = home / klayout_folder / "pymacros"
    dest_folder.mkdir(exist_ok=True, parents=True)
    cwd = pathlib.Path(__file__).resolve().parent
    src = cwd / "klayout" / "pymacros" / "klive.lym"
    dest = dest_folder / "klive.lym"

    if dest.exists():
        print(f"removing klive already installed in {dest}")
        os.remove(dest)

    shutil.copy(src, dest)
    print(f"klive installed to {dest}")


def copy(src, dest):
    """overwrite file or directory"""
    dest_folder = dest.parent
    dest_folder.mkdir(exist_ok=True, parents=True)

    if dest.exists():
        print(f"removing {dest} already installed")
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            os.remove(dest)

    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy(src, dest)
    print(f"{src} copied to {dest}")


def install_generic_tech():
    if sys.platform == "win32":
        klayout_folder = "KLayout"
    else:
        klayout_folder = ".klayout"

    cwd = pathlib.Path(__file__).resolve().parent
    home = pathlib.Path.home()
    src = cwd / "klayout" / "tech"
    dest = home / klayout_folder / "tech" / "generic"

    copy(src, dest)

    src = cwd / "klayout" / "drc" / "generic.lydrc"
    dest = home / klayout_folder / "drc" / "generic.lydrc"
    copy(src, dest)


if __name__ == "__main__":
    cwd = pathlib.Path(__file__).resolve().parent
    home = pathlib.Path.home()
    src = cwd / "klayout" / "tech"

    install_gdsdiff()
    install_klive()
    install_generic_tech()
