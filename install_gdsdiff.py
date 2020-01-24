import pathlib
import configparser

home = pathlib.Path.home()
git_config_path = home / ".gitconfig"
git_attributes_path = home / ".gitattributes"


git_config_str = open(git_config_path).read()
git_attributes_str = open(git_attributes_path).read()

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

        with open(git_config_path, "w") as f:
            config.write(f, space_around_delimiters=True)

if "gds_diff" not in git_attributes_str:
    print("Appending the gdsdiff command to your ~/.gitattributes")

    with open(git_attributes_path, "a") as f:
        f.write("*.gds diff=gds_diff\n")
