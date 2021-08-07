from pathlib import Path
from typing import Tuple

import gdsfactory as gf
from gdsfactory.mask.merge_json import merge_json
from gdsfactory.mask.merge_markdown import merge_markdown
from gdsfactory.mask.merge_test_metadata import merge_test_metadata
from gdsfactory.mask.write_labels import write_labels


def merge_metadata(
    gdspath: Path,
    labels_prefix: str = "opt",
    label_layer: Tuple[int, int] = gf.LAYER.LABEL,
    **kwargs
) -> None:
    """Merges all JSON metadata into a big JSON.

    Args:
        gdspath: GDSpath
        labels_prefix
        label_layer: layer for the labels
    """
    mdpath = gdspath.with_suffix(".md")
    jsonpath = gdspath.with_suffix(".json")

    build_directory = gdspath.parent.parent
    doe_directory = build_directory / "doe"

    write_labels(gdspath=gdspath, prefix=labels_prefix, label_layer=label_layer)

    merge_json(doe_directory=doe_directory, jsonpath=jsonpath, **kwargs)
    merge_markdown(reports_directory=doe_directory, mdpath=mdpath)
    merge_test_metadata(gdspath, labels_prefix=labels_prefix)


if __name__ == "__main__":

    gdspath = gf.CONFIG["samples_path"] / "mask" / "build" / "mask" / "mask.gds"
    print(gdspath)
    merge_metadata(gdspath)
