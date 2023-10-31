"""Converts CSV of test site labels into a CSV test manifest."""

import json
import pathlib
from functools import partial

import pandas as pd

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle

marker = partial(gf.components.rectangle, layer="TEXT", centered=True, size=(10, 10))


def write_test_manifest(
    csvpath: str | pathlib.Path,
    gdspath: str | pathlib.Path | None = None,
    marker_optical=marker,
    marker_electrical=marker,
) -> pd.DataFrame:
    """Converts CSV of test site labels into a CSV test manifest.

    Args:
        csvpath: path to CSV file with test site labels.
        gdspath: path to GDS file with test site labels.
        marker_optical: marker to use for test site labels.
        marker_electrical: marker to use for test site labels.
    """
    df_in = pd.read_csv(csvpath)

    # Initialize an empty list to collect the rows
    rows = []
    columns = [
        "cell",
        "device",
        "port_type",
        "x",
        "y",
        "measurement",
        "measurement_settings",
        "doe",
        "analysis",
        "analysis_settings",
        "cell_settings",
    ]
    if gdspath:
        c = gf.Component()
        _ = c << gf.import_gds(gdspath)
        marker_optical = marker_optical()
        marker_electrical = marker_electrical()

    for _, label in df_in.iterrows():
        x, y, _orientation = label.x, label.y, label.rotation
        text = label.text
        d = json.loads(text)

        if gdspath:
            if d["port_type"] == "optical":
                ref = c << marker_optical
                ref.x = x
                ref.y = y
            elif d["port_type"] == "electrical":
                ref = c << marker_electrical
                ref.x = x
                ref.y = y

        row = [
            d["name"],
            d["name"] + f"_{int(x)}_{int(y)}",
            d["port_type"],
            x,
            y,
            d["measurement"],
            d["measurement_settings"],
            d["doe"],
            d["analysis"],
            d["analysis_settings"],
            d["cell_settings"],
        ]
        rows.append(row)

    if gdspath:
        c.show()

    return pd.DataFrame(
        rows,
        columns=columns,
    )


if __name__ == "__main__":
    c = sample_reticle(grid=False)
    # c = c.mirror()
    c.show(show_ports=False)
    gdspath = c.write_gds("sample_reticle.gds")
    csvpath = gf.labels.write_labels.write_labels_gdstk(
        gdspath, prefixes=("{",), layer_label="TEXT"
    )
    df = write_test_manifest(csvpath, gdspath=gdspath)
    df.to_csv("test_manifest.csv")
    print(df)
