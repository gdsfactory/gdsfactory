import os
from glob import glob

from pp.config import logging, load_config, CONFIG


def merge_markdown(config=CONFIG):
    """ Merges all individual markdown reports (.md) into a single markdown
    you can add a report:[Capacitors, Diodes...] in config.yml to define the merge order
    """
    mask_name = config["mask"]["name"]
    reports_directory = config["gds_directory"]
    report_path = config["mask_directory"] / (mask_name + ".md")

    with open(report_path, "w") as f:

        def wl(line="", eol="\n"):
            f.write(line + eol)

        doe_names_list = CONFIG.get("report")
        """ check if reports follows a particular order """
        if doe_names_list:
            for doe_name in doe_names_list:
                filename = os.path.join(reports_directory, doe_name + ".md")
                with open(filename) as infile:
                    for line in infile:
                        f.write(line)

        else:
            reports = sorted(glob(os.path.join(reports_directory, "*.md")))
            for filename in reports:
                with open(filename) as infile:
                    for line in infile:
                        f.write(line)

    logging.info("Wrote {}".format(os.path.relpath(report_path)))


if __name__ == "__main__":
    config_path = CONFIG["samples_path"] / "mask" / "config.yml"
    config = load_config(config_path)
    merge_markdown(config)

    # print(config['gds_directory'])
    # from pprint import pprint
    # pprint(config)
