"""Combine multiple YAML files into one."""

from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from gdsfactory.config import logger
from gdsfactory.types import PathType


def merge_yaml(
    doe_directory: PathType,
    yaml_path: Optional[PathType] = None,
    json_version: int = 6,
) -> Dict[str, Any]:
    """Combine several YAML files

    in the root of the mask directory, gets mask_name from there

    Args:
        doe_directory: defaults to current working directory
        extra_directories: list of extra_directories
        yaml_path: optional metadata path to write metadata
        json_version:

    """
    logger.debug(f"Merging JSON files from {doe_directory}")
    cells = {}

    for filename in doe_directory.glob("**/*.yml"):
        logger.debug(f"merging {filename}")
        metadata = OmegaConf.load(filename)
        metadata = OmegaConf.to_container(metadata)
        cells.update(metadata.get("cells"))

    metadata = dict(
        json_version=json_version,
        cells=cells,
    )

    if yaml_path:
        yaml_path.write_text(OmegaConf.to_yaml(metadata))
        logger.info(f"Wrote metadata in {yaml_path}")
    return metadata


if __name__ == "__main__":
    from pprint import pprint

    import gdsfactory as gf

    gdspath = (
        gf.CONFIG["samples_path"] / "mask_custom" / "build" / "mask" / "sample_mask.gds"
    )
    build_directory = gdspath.parent.parent
    doe_directory = build_directory / "cache_doe"
    yaml_path = gdspath.with_suffix(".yml")

    d = merge_yaml(doe_directory=doe_directory, yaml_path=yaml_path)
    pprint(d)
