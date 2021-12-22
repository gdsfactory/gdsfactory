"""Merge mask metadata with test labels to return test_metadata """

import pathlib
import warnings
from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf

from gdsfactory.config import CONFIG, logger


def parse_csv_data(csv_labels_path: Path) -> List[List[str]]:
    """Returns CSV labels."""
    with open(csv_labels_path) as f:
        # Get all lines
        lines = [line.replace("\n", "") for line in f.readlines()]

        # Ignore labels for metrology structures
        lines = [line for line in lines if not line.startswith("METR_")]

        # Split lines in fields
        lines = [line.split(",") for line in lines]

        lines = [[s.strip() for s in splitted if s.strip()] for splitted in lines]

        # Remove empty lines
        lines = [line for line in lines if line]
    return lines


def get_cell_from_label(label: str) -> str:
    """get cell name from the label (cell_name is in parenthesis)"""
    cell_name = label.split("(")[1].split(")")[0]
    if cell_name.startswith("loopback"):
        cell_name = "_".join(cell_name.split("_")[1:])
    return cell_name


def merge_test_metadata(
    gdspath: Path = CONFIG["mask_gds"], labels_prefix: str = "opt"
) -> DictConfig:
    """Returns a test metadata dict config of labeled cells
    by merging GDS labels in CSV and YAML mask metadata

    Args:
        gdspath: for GDS file
        labels_prefix: only select labels with a text prefix

    .. code::

        CSV labels  -------
                          |--> merge_test_metadata dict
                          |
        YAML metatada  ----


    """
    gdspath = pathlib.Path(gdspath)
    mask_metadata_path = gdspath.with_suffix(".yml")
    csv_labels_path = gdspath.with_suffix(".csv")
    test_metadata_path = gdspath.with_suffix(".tp.yml")

    if not mask_metadata_path.exists():
        raise FileNotFoundError(f"missing mask YAML metadata {mask_metadata_path}")

    if not csv_labels_path.exists():
        raise FileNotFoundError(f"missing CSV labels {csv_labels_path}")

    labels_list = parse_csv_data(csv_labels_path)
    metadata = OmegaConf.load(mask_metadata_path)
    cells_metadata = metadata.get("cells", {})

    test_metadata = DictConfig({})

    for label, x, y in labels_list:
        cell = get_cell_from_label(label)

        if cell in cells_metadata:
            test_metadata[cell] = cells_metadata[cell]
            test_metadata[cell].label = dict(x=x, y=y, text=label)
        else:
            logger.error(f"missing cell metadata for {cell}")
            warnings.warn(f"missing cell metadata for {cell}")

    OmegaConf.save(test_metadata, f=test_metadata_path)
    return test_metadata


if __name__ == "__main__":
    from gdsfactory import CONFIG

    # gdspath = CONFIG["repo_path"] / "samples" / "mask" / "build" / "mask" / "mask.gds"
    gdspath = (
        CONFIG["samples_path"] / "mask_pack" / "build" / "mask" / "sample_mask.gds"
    )
    d = merge_test_metadata(gdspath)
    print(d)
