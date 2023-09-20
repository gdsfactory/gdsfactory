"""Converts CSV of test site labels into a CSV test manifest."""

import json
import pathlib

import numpy as np
import pandas as pd

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle


def write_test_manifest(csvpath: str | pathlib.Path) -> pd.DataFrame:
    """Converts CSV of test site labels into a CSV test manifest."""
    df_in = pd.read_csv(csvpath)

    # Initialize an empty list to collect the rows
    rows = []
    columns = [
        "name",
        "xopt",
        "yopt",
        "xelec",
        "yelec",
        "measurement",
        "measurement_settings",
        "doe",
        "analysis",
        "analysis_settings",
    ]

    for _, label in df_in.iterrows():
        x, y, _orientation = label.x, label.y, label.rotation
        text = label.text
        d = json.loads(text)

        row = [
            d["name"] + f"_{int(x)}_{int(y)}",
            [np.round(i + x, 3) for i in d["xopt"]],
            [np.round(i + y, 3) for i in d["yopt"]],
            [np.round(i + x, 3) for i in d["xelec"]],
            [np.round(i + y, 3) for i in d["yelec"]],
            d["measurement"],
            d["measurement_settings"],
            d["doe"],
            d["analysis"],
            d["analysis_settings"],
        ]
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=columns,
    )


if __name__ == "__main__":
    c = sample_reticle(grid=False)
    c.show(show_ports=True)
    gdspath = c.write_gds()
    csvpath = gf.labels.write_labels.write_labels_gdstk(
        gdspath, prefixes=("{",), layer_label="TEXT"
    )
    df = write_test_manifest(csvpath)
    df.to_csv("test_manifest.csv")
    print(df)
