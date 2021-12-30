"""Merge mask metadata with test labels to return test_metadata """
import warnings
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import DictConfig

from gdsfactory.config import logger
from gdsfactory.types import PathType


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
    labels_path: PathType,
    mask_metadata: Dict[str, Any],
    labels_prefix: str = "opt",
) -> DictConfig:
    """Returns a test metadata dict config of labeled cells
    by merging GDS labels in CSV and YAML mask metadata

    Args:
        labels_path: for test labels in CSV
        mask_metadata: dict with test metadata
        labels_prefix: only select labels with a text prefix

    .. code::

        CSV labels  -------
                          |--> merge_test_metadata dict
                          |
        YAML metatada  ----


    """
    labels_path = Path(labels_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"missing CSV labels {labels_path}")

    labels_list = parse_csv_data(labels_path)
    cells_metadata = mask_metadata.get("cells", {})

    test_metadata = DictConfig({})

    for label, x, y in labels_list:
        cell = get_cell_from_label(label)

        if cell in cells_metadata:
            test_metadata[cell] = cells_metadata[cell]
            test_metadata[cell].label = dict(x=float(x), y=float(y), text=label)
        else:
            logger.error(f"missing cell metadata for {cell}")
            warnings.warn(f"missing cell metadata for {cell}")

    return test_metadata


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from gdsfactory import CONFIG

    labels_path = (
        CONFIG["samples_path"] / "mask_pack" / "build" / "mask" / "sample_mask.csv"
    )
    mask_metadata_path = labels_path.with_suffix(".yml")
    mask_metadata = OmegaConf.load(mask_metadata_path)
    d = merge_test_metadata(labels_path=labels_path, mask_metadata=mask_metadata)
    print(d)
