"""Converts CSV of test site labels into a CSV test manifest."""

import pathlib
import warnings

import gdsfactory as gf
from gdsfactory.labels.write_test_manifest import write_test_manifest


def get_test_manifest(component: gf.Component, csvpath: str | pathlib.Path, **kwargs):
    """Returns a pandas DataFrame with test manifest."""
    warnings.warn(
        "get_test_manifest is deprecated, use write_test_manifest instead",
        DeprecationWarning,
    )
    import pandas as pd

    write_test_manifest(component, csvpath, **kwargs)
    return pd.read_csv(csvpath)


if __name__ == "__main__":
    from gdsfactory.samples.sample_reticle import sample_reticle

    c = sample_reticle()
    # c = gf.pack([c])[0]
    c.show()
    df = get_test_manifest(c)
    df.to_csv("test_manifest.csv", index=False)
    print(df)
