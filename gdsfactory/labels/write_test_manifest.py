"""Converts CSV of test site labels into a CSV test manifest."""

import json
import pathlib
import warnings
from functools import partial

import pandas as pd

import gdsfactory as gf

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

    warnings.warn(
        "This function is deprecated. Use gf.labels.get_test_manifest instead",
        DeprecationWarning,
    )

    df_in = pd.read_csv(csvpath)

    # Initialize an empty list to collect the rows
    rows = []
    columns = [
        "cell",
        "device",
        "port_type",
        "port_names",
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
            d["port_names"],
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
    test_info_mzi_heaters = dict(
        doe="mzis_heaters",
        analysis="mzi_heater_phase_shifter_length",
        measurement="optical_loopback4_heater_sweep",
    )
    test_info_ring_heaters = dict(
        doe="ring_heaters_coupling_length",
        analysis="ring_heater",
        measurement="optical_loopback2_heater_sweep",
    )

    mzis = [
        gf.components.mzi2x2_2x2_phase_shifter(length_x=lengths)
        for lengths in [100, 200, 300]
    ]

    rings = [
        gf.components.ring_single_heater(length_x=length_x) for length_x in [10, 20, 30]
    ]

    mzis_te = [
        gf.components.add_fiber_array_optical_south_electrical_north(
            mzi,
            electrical_port_names=["top_l_e2", "top_r_e2"],
            info=test_info_mzi_heaters,
            decorator=gf.labels.add_label_json,
        )
        for mzi in mzis
    ]
    rings_te = [
        gf.components.add_fiber_array_optical_south_electrical_north(
            ring,
            electrical_port_names=["l_e2", "r_e2"],
            info=test_info_ring_heaters,
            decorator=gf.labels.add_label_json,
        )
        for ring in rings
    ]
    c = gf.pack(mzis_te + rings_te)[0]
    c.show(show_ports=False)
    gdspath = c.write_gds("sample_reticle.gds")
    csvpath = gf.labels.write_labels.write_labels_gdstk(
        gdspath, prefixes=("{",), layer_label="TEXT"
    )
    df = write_test_manifest(csvpath, gdspath=gdspath)
    df.to_csv("test_manifest.csv")
    print(df)
