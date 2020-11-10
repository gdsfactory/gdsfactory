import os
import shutil
import sys
import configparser
import pathlib
from setuptools import find_packages, setup
from setuptools.command.install import install


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


def symlink(src, dest):
    """ installs generic layermap """
    if dest.exists():
        print("generic tech already installed")
        return

    dest_folder = dest.parent
    dest_folder.mkdir(exist_ok=True, parents=True)
    try:
        os.symlink(src, dest)
    except Exception:
        os.remove(dest)
        os.symlink(src, dest)
    print(f"added symlink from {src} to {dest}")


def install_generic_tech():
    if sys.platform == "win32":
        klayout_folder = "KLayout"
    else:
        klayout_folder = ".klayout"

    cwd = pathlib.Path(__file__).resolve().parent
    home = pathlib.Path.home()
    src = cwd / "klayout" / "tech"
    dest = home / klayout_folder / "tech" / "generic"

    symlink(src, dest)

    src = cwd / "klayout" / "drc" / "generic.lydrc"
    dest = home / klayout_folder / "drc" / "generic.lydrc"
    symlink(src, dest)


def get_install_requires():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if not line.startswith("-")]


class CustomInstallCommand(install):
    """Customized setuptools install command """

    def run(self):
        install_klive()
        install_gdsdiff()
        install_generic_tech()
        install.run(self)


setup(
    name="gdsfactory",
    url="https://github.com/gdsfactory/gdsfactory",
    version="2.1.2",
    author="PsiQ",
    scripts=["pp/pf.py"],
    description="python libraries to generate GDS layouts",
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_install_requires(),
    python_requires=">=3.6",
    cmdclass={"install": CustomInstallCommand},
    entry_points="""
        [console_scripts]
        pf=pp.pf:cli
    """,
)
