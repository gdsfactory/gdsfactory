import os
from glob import glob

from omegaconf import OmegaConf

from pp.config import CONFIG, conf, logging


def merge_markdown(
    reports_directory=CONFIG["doe_directory"],
    mdpath=CONFIG["mask_directory"] / "report.md",
    **kwargs,
):
    """Merges all individual markdown reports (.md) into a single markdown
    you can add a report:[Capacitors, Diodes...] in config.yml to define the merge order
    """
    logging.debug("Merging Markdown files:")
    configpath = mdpath.with_suffix(".yml")

    with open(configpath, "w") as f:
        conf.update(**kwargs)
        f.write(OmegaConf.to_yaml(conf))

    with open(mdpath, "w") as f:

        def wl(line="", eol="\n"):
            f.write(line + eol)

        reports = sorted(glob(os.path.join(reports_directory, "*.md")))
        for filename in reports:
            with open(filename) as infile:
                for line in infile:
                    f.write(line)

    logging.info(f"Wrote {mdpath}")
    logging.info(f"Wrote {configpath}")


if __name__ == "__main__":
    reports_directory = CONFIG["samples_path"] / "mask" / "does"
    merge_markdown(reports_directory)
