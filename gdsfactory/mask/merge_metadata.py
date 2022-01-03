from pathlib import Path
from typing import Tuple

from omegaconf import DictConfig, OmegaConf

import gdsfactory as gf
from gdsfactory.mask.merge_markdown import merge_markdown
from gdsfactory.mask.merge_test_metadata import merge_test_metadata

# from gdsfactory.mask.merge_json import merge_json
from gdsfactory.mask.merge_yaml import merge_yaml
from gdsfactory.mask.write_labels import write_labels


def merge_metadata(
    gdspath: Path,
    labels_prefix: str = "opt",
    layer_label: Tuple[int, int] = gf.LAYER.TEXT,
) -> DictConfig:
    """Merges all mask metadata and returns test metadata
    This function works well only when you define the mask in YAML
    Exports a YAML file with only the cells information that have a valid test and measurement label

    For the cells that need to be measure we add test labels

    This is the automatic version of write_labels combined with merge_test_metadata

    .. code::

        CSV labels  -------
                          |--> merge_test_metadata dict
                          |
        YAML metatada  ----


    Args:
        gdspath: GDSpath
        labels_prefix
        layer_label: layer for the labels
    """
    mdpath = gdspath.with_suffix(".md")
    yaml_path = gdspath.with_suffix(".yml")
    test_metadata_path = gdspath.with_suffix(".tp.yml")

    build_directory = gdspath.parent.parent
    doe_directory = build_directory / "cache_doe"

    labels_path = write_labels(
        gdspath=gdspath, prefix=labels_prefix, layer_label=layer_label
    )

    # jsonpath = gdspath.with_suffix(".yml")
    # merge_json(doe_directory=doe_directory, jsonpath=jsonpath, **kwargs)

    mask_metadata = merge_yaml(doe_directory=doe_directory, yaml_path=yaml_path)
    merge_markdown(reports_directory=doe_directory, mdpath=mdpath)
    tm = merge_test_metadata(
        labels_prefix=labels_prefix,
        mask_metadata=mask_metadata,
        labels_path=labels_path,
    )

    test_metadata_path.write_text(OmegaConf.to_yaml(tm))
    return tm


if __name__ == "__main__":
    gdspath = (
        gf.CONFIG["samples_path"] / "mask_custom" / "build" / "mask" / "sample_mask.gds"
    )
    tm = merge_metadata(gdspath)
