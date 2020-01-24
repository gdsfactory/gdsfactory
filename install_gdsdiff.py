import pathlib
import configparser

home = pathlib.Path.home()
config_path = home / ".gitconfig"

cwd = pathlib.Path(__file__).resolve()
src = cwd / "gdsdiff" / ".gitattributes"


gitattr = 2


config_str = open(config_path).read()

if "gdsdiff" not in config_str:
    print("gdsdiff shows boolean differences in Klayout")
    print("git diff FILE.GDS")
    print("We need to append the gdsdiff command to your ~/.gitattributes")
    answer = input("are you ok appending gdsdiff to your .gitattributes? (y/n)")

    if answer.lower().startswith("y"):

        config = configparser.RawConfigParser()
        config.read(config_path)
        key = 'diff "gds_diff"'

        if key not in config.sections():
            config.add_section(key)
            config.set(key, "command", "python -m gdsdiff.gds_diff_git")
            config.set(key, "binary", "True")

            with open(config_path, "w") as f:
                print("installing gds_diff")
                config.write(f, space_around_delimiters=True)
