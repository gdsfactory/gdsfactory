"""Merge mask metadata with test labels to return test_metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from gdsfactory.config import logger
from gdsfactory.typings import PathType


def parse_csv_data(
    csv_labels_path: Path, ignore_prefix: str = "METR_"
) -> List[List[str]]:
    """Returns CSV labels as a list of strings."""
    with open(csv_labels_path) as f:
        # Get all lines
        lines = [line.replace("\n", "") for line in f.readlines()]

        # Ignore labels for metrology structures
        lines = [line for line in lines if not line.startswith(ignore_prefix)]

        # Split lines in fields
        lines = [line.split(",") for line in lines]

        lines = [[s.strip() for s in split if s.strip()] for split in lines]

        # Remove empty lines
        lines = [line for line in lines if line]
    return lines


def get_cell_from_label_brackets(label: str) -> str:
    """Get cell name from the label (cell_name is in parenthesis)."""
    try:
        cell_name = label.split("(")[1].split(")")[0]
    except IndexError as error:
        raise ValueError(f"{label!r} needs (cell name) between parenthesis") from error

    if cell_name.startswith("loopback"):
        cell_name = "_".join(cell_name.split("_")[1:])
    return cell_name


def get_cell_from_label(label: str) -> str:
    """Returns label, assuming opt-GratingName-ComponentName-PortName"""

    if label.startswith("elec"):
        if not label.split("-"):
            raise ValueError(f"{label!r} needs to follow elec-ComponentName-PortName")
        return label.split("-")[1]

    if len(label.split("-")) < 2:
        raise ValueError(
            f"{label!r} needs to follow opt-GratingName-ComponentName-PortName"
        )
    return label.split("-")[2]


def merge_test_metadata(
    labels_path: PathType,
    mask_metadata: Dict[str, Any],
    get_cell_from_string=get_cell_from_label,
    filepath: Optional[PathType] = None,
) -> DictConfig:
    """Returns a test metadata dict config of labeled cells by merging GDS \
    labels in CSV and YAML mask metadata.

    Args:
        labels_path: for test labels in CSV.
        mask_metadata: dict with test metadata.
        get_cell_from_string: returns label string.
        filepath: Optional path to write test metadata.

    .. code::

        CSV labels  -------
                          |--> merge_test_metadata dict
                          |
        YAML metadata  ----

    """
    labels_path = Path(labels_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"missing CSV labels {labels_path!r}")

    labels_list = parse_csv_data(labels_path)
    cells_metadata = mask_metadata.get("cells", {})

    test_metadata = DictConfig({})

    for label, x, y, angle in labels_list[1:]:
        cell = get_cell_from_string(label)

        if cell in cells_metadata:
            test_metadata[cell] = cells_metadata[cell]
            test_metadata[cell].label = dict(
                x=float(x), y=float(y), text=label, angle=angle
            )
        else:
            logger.warning(f"missing cell metadata for {cell!r}")

    if filepath:
        filepath = Path(filepath)
        filepath.write_text(OmegaConf.to_yaml(test_metadata))

    return test_metadata


if __name__ == "__main__":
    # from gdsfactory import PATH

    # labels_path = (
    #    PATH.samples_path / "mask_pack" / "build" / "mask" / "sample_mask.csv"
    # )
    # mask_metadata_path = labels_path.with_suffix(".yml")
    # mask_metadata = OmegaConf.load(mask_metadata_path)
    # d = merge_test_metadata(labels_path=labels_path, mask_metadata=mask_metadata)
    # print(d)
    print(get_cell_from_label("opt_te1550_demo"))
