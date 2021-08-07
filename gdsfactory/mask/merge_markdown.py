import dataclasses
import os
from glob import glob
from pathlib import Path

from omegaconf import OmegaConf

from gdsfactory.config import CONFIG, TECH, logger


def merge_markdown(
    reports_directory: Path = CONFIG["doe_directory"],
    mdpath: Path = CONFIG["mask_directory"] / "report.md",
    **kwargs,
) -> None:
    """Merges all individual markdown reports (.md) into a single markdown
    you can add a report:[Capacitors, Diodes...] in config.yml to define the merge order
    """
    logger.info("Merging Markdown files:")
    configpath = mdpath.with_suffix(".yml")
    tech = dataclasses.asdict(TECH)
    tech.pop("library", "")

    with open(configpath, "w") as f:
        tech.update(**kwargs)
        tech_omegaconf = OmegaConf.create(tech)
        f.write(OmegaConf.to_yaml(tech_omegaconf))

    with open(mdpath, "w") as f:

        def wl(line="", eol="\n"):
            f.write(line + eol)

        reports = sorted(glob(os.path.join(reports_directory, "*.md")))
        for filename in reports:
            with open(filename) as infile:
                for line in infile:
                    f.write(line)

    logger.info(f"Wrote {mdpath}")
    logger.info(f"Wrote {configpath}")


if __name__ == "__main__":
    reports_directory = CONFIG["samples_path"] / "mask" / "does"
    merge_markdown(reports_directory)
