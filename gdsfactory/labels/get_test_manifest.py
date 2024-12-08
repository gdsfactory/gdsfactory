"""Converts CSV of test site labels into a CSV test manifest."""

import pathlib
from typing import Any

import pandas as pd  # type: ignore

import gdsfactory as gf
from gdsfactory._deprecation import deprecate
from gdsfactory.labels.write_test_manifest import write_test_manifest


def get_test_manifest(
    component: gf.Component, csvpath: str | pathlib.Path, **kwargs: Any
) -> pd.DataFrame:
    """Returns a pandas DataFrame with test manifest."""
    deprecate("get_test_manifest", "write_test_manifest")

    write_test_manifest(component, csvpath, **kwargs)
    return pd.read_csv(csvpath)  # type: ignore


if __name__ == "__main__":
    from gdsfactory.samples.sample_reticle import sample_reticle

    c = sample_reticle()
    # c = gf.pack([c])[0]
    c.show()
    df = get_test_manifest(c)  # type: ignore
    df.to_csv("test_manifest.csv", index=False)  # type: ignore
    print(df)  # type: ignore
